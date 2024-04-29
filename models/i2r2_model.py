import torch
import torch.nn as nn
from torch.nn import functional as F
from models import common, swt_pytorch
from torch.autograd import Variable
from torch.nn import init

#def make_model(args, parent=False):
#    return Rainnet(args)


def get_residue(tensor, r_dim=1):
    """
    return residue_channel
    """
    max_channel = torch.max(tensor, dim=r_dim, keepdim=True)  # keepdim
    min_channel = torch.min(tensor, dim=r_dim, keepdim=True)

    res_channel = max_channel[0] - min_channel[0]  # Ablation
    res_channel_cate = torch.cat((res_channel, res_channel, res_channel), dim=1)

    return res_channel_cate


def wave (t1, t2, t3):
    """
    Separate
    """
    # res_channel = []
    t1_0, t1_1, t1_2, t1_3 = torch.chunk(t1[0], 4, dim=1)
    t2_0, t2_1, t2_2, t2_3 = torch.chunk(t2[0], 4, dim=1)
    t3_0, t3_1, t3_2, t3_3 = torch.chunk(t3[0], 4, dim=1)
    ll = torch.cat((t1_0, t2_0, t3_0), dim=1)  # keepdim
    lh = torch.cat((t1_1, t2_1, t3_1), dim=1)
    hl = torch.cat((t1_2, t2_2, t3_2), dim=1)
    hh = torch.cat((t1_3, t2_3, t3_3), dim=1)
    total = torch.cat((ll, lh, hl, hh), dim=1)

    return total   # (RCP module) -> remove rain streak & get background detail


def get_list(out):

    SWT_list = [out]

    return SWT_list


def find_LL(out):
    yl = out[:, 0:3, :, :]
    return yl


class convd(nn.Module):
    def __init__(self, inputchannel, outchannel, kernel_size, stride):
        super(convd, self).__init__()
        self.relu = nn.ReLU()
        self.padding = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(inputchannel, outchannel, kernel_size, stride)

    def forward(self, x):
        x = self.conv(self.padding(x))
        x = self.relu(x)
        return x  # normal Conv block


class RB(nn.Module):
    """
    SE_Resblock
    - Compose of two Conv+ReLU block and one SE_block
    """
    def __init__(self, n_feats, nm='in', use_GPU=True):
        super(RB, self).__init__()

        module_body = []
        for i in range(2):  # conv-ReLU
            module_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
            module_body.append(nn.ReLU())

        self.module_body = nn.Sequential(*module_body)
        self.relu = nn.ReLU()
        self.se = common.SELayer(n_feats, 1)

    def forward(self, x):
        res = self.module_body(x)
        res = self.se(res)
        res += x
        return res


class RIR_LSTM(nn.Module):
    """
    ResLSTM
    - Mainly used to extract the feature of the image
    - compose of one ConvLSTM & two SE_Block
    """
    def __init__(self, n_feats, n_blocks, nm='in', recurrent_iter=3, use_GPU=True):  # recurrent_iter 값 받아오기   # iter 증가

        super(RIR_LSTM, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv_rb1 = RB(n_feats)  # 수정
        self.conv_rb2 = RB(n_feats)

        self.conv_i = nn.Sequential(
            nn.Conv2d(n_feats + n_feats, n_feats, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(n_feats + n_feats, n_feats, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(n_feats + n_feats, n_feats, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(n_feats + n_feats, n_feats, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h

            x = self.conv_rb1(x)  # (conv+relu)x2 + SE
            x = self.conv_rb2(x)
            x = self.conv(x)
            x = x + input
            x_list.append(x)

        return x, x_list


class res_ch(nn.Module):
    """
    RCP feature extraction module(in paper ResLSTM 1)
    - To extract the feature from RCP
    - Compose of two Conv+RELU block and one ResLSTM block
    """
    def __init__(self, n_feats, blocks=1, softsign=False):
        super(res_ch, self).__init__()
        self.conv_init1 = convd(3, n_feats // 2, 3, 1)
        self.conv_init2 = convd(n_feats // 2, n_feats, 3, 1)
        self.extra = RIR_LSTM(n_feats, n_blocks=blocks)

    def forward(self, x):
        x = self.conv_init2(self.conv_init1(x))
        x, _ = self.extra(x)
        return x


class Prior_Sp(nn.Module):
    """
    Fusion Block
    - Fuse the RCP features and Wavelet Features
    - emphasize the importance of the feature
    """
    def __init__(self, in_dim=32):
        super(Prior_Sp, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.gamma1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.gamma2 = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.se = common.SELayer(channel=32, reduction=8)  # SE block to fuse RCPs

    def forward(self, x, prior):
        x_q = self.query_conv(x)
        prior_k = self.key_conv(prior)
        energy = x_q * prior_k
        se = self.se(energy)

        x_gamma = self.gamma1(se)
        x_out = x + x_gamma

        p_gamma = self.gamma2(se)
        prior_out = prior + p_gamma

        return x_out, prior_out


class Feature_ex(nn.Module):
    """
    ResLSTM_4
    - Use for main network feature extraction
    - Compose of four ResLSTM block
    - Update the information pass throw each block
    """
    def __init__(self, n_feats, blocks=1):  # block
        super(Feature_ex, self).__init__()

        # fuse res
        self.prior = Prior_Sp()
        self.fuse_res = convd(n_feats * 2, n_feats, 3, 1)

        self.branch1_1 = RIR_LSTM(n_feats=n_feats, n_blocks=blocks)
        self.branch1_2 = RIR_LSTM(n_feats=n_feats, n_blocks=blocks)
        self.branch1_3 = RIR_LSTM(n_feats=n_feats, n_blocks=blocks)
        self.branch1_4 = RIR_LSTM(n_feats=n_feats, n_blocks=blocks)

    def forward(self, x, res_feats):
        x_p, res_feats_p = self.prior(x, res_feats)
        x_s = torch.cat((x_p, res_feats_p), dim=1)

        x1_init = self.fuse_res(x_s)

        # ResLSTM_1
        x1_1, _ = self.branch1_1(x1_init)
        x1_i = x1_1

        # ResLSTM_2
        x1_2, _ = self.branch1_2(x1_i)
        x1_i = x1_2

        # ResLSTM_3
        x1_3, _ = self.branch1_3(x1_i)
        x1_i = x1_3

        # ResLSTM_4
        x1_4, _ = self.branch1_4(x1_i)
        x1_i = x1_4

        return x1_i  # feature map


"""
    Main
"""


class Rainnet(nn.Module):
    def __init__(self):
        super(Rainnet, self).__init__()
        n_feats = 32
        blocks = 3

        self.SWT = swt_pytorch.SWTForward(J=1, wave='haar', mode='zero').cuda()
        self.ISWT = swt_pytorch.SWTInverse(wave='haar', mode='zero').cuda()  # definition about Wavelet Transform

        self.conv_init1 = convd(12, n_feats // 2, 3, 1)
        self.conv_init2 = convd(n_feats // 2, n_feats, 3, 1)
        self.res_extra1 = res_ch(n_feats, blocks)
        self.sub1 = Feature_ex(n_feats, blocks)
        self.res_extra2 = res_ch(n_feats, blocks)
        self.sub2 = Feature_ex(n_feats, blocks)
        self.res_extra3 = res_ch(n_feats, blocks)
        self.sub3 = Feature_ex(n_feats, blocks)

        self.ag1 = convd(n_feats * 2, n_feats, 3, 1)
        self.ag2 = convd(n_feats * 3, n_feats, 3, 1)
        self.ag2_en = convd(n_feats * 2, n_feats, 3, 1)
        self.ag_en = convd(n_feats * 3, n_feats, 3, 1)

        self.output1 = nn.Conv2d(n_feats, 12, 3, 1, padding=1)
        self.output2 = nn.Conv2d(n_feats, 12, 3, 1, padding=1)
        self.output3 = nn.Conv2d(n_feats, 12, 3, 1, padding=1)

    def forward(self, x):

        r, g, b = torch.split(x, 1, dim=1)

        red = self.SWT(r)
        green = self.SWT(g)
        blue = self.SWT(b)
        """""""""""""""""
        #### Stage 1 ####
        """""""""""""""""
        x0 = wave(red, green, blue)
        res_rgb0 = get_residue(x)

        x_init = self.conv_init2(self.conv_init1(x0))
        x1 = self.sub1(x_init, self.res_extra1(res_rgb0))
        out1_b = self.output1(x1)
        out1_ = get_list(out1_b)

        out1 = self.ISWT(out1_)
        """""""""""""""""
        #### Stage 2 ####
        """""""""""""""""
        x1_ll = find_LL(out1_b)
        res_ll1 = get_residue(x1_ll)

        x2 = self.sub2(self.ag1(torch.cat((x1, x_init), dim=1)),
                       self.res_extra2(res_ll1))  # + x1 # 2  # 수정
        x2_ = self.ag2_en(torch.cat([x2, x1], dim=1))
        out2_b = self.output2(x2_)
        out2_ = get_list(out2_b)

        out2 = self.ISWT(out2_)
        """""""""""""""""
        #### Stage 3 ####
        """""""""""""""""
        x2_ll = find_LL(out2_b)
        res_ll2 = get_residue(x2_ll)

        x3 = self.sub3(self.ag2(torch.cat((x2, x1, x_init), dim=1)),
                       self.res_extra3(res_ll2))  # + x2 # 3 # 수정
        x3 = self.ag_en(torch.cat([x3, x2, x1], dim=1))
        out3_b = self.output3(x3)
        out3_ = get_list(out3_b)
        out3 = self.ISWT(out3_)

        return out3, out2, out1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def build_net():
    return Rainnet()