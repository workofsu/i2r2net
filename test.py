import os
import torch
import argparse
from models.i2r2_model import build_net
from utils import Adder, Tensor2_np
import time
from torchvision.transforms import functional as F
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as f


class DerainDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'rain/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'rain', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'norain', self.image_list[idx]))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError


def test_dataloader(path, batch_size=1, num_workers=1):
    dataloader = DataLoader(
        DerainDataset(path, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def print_network_parameter(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)


parser = argparse.ArgumentParser()

# Directories
parser.add_argument('--model_name', default='Rainnet', type=str)
parser.add_argument('--data_dir', type=str, default='....test_image data path....')
parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])
parser.add_argument('--save_file_name', default='save_file_path', type=str)
parser.add_argument('--num_worker', type=int, default=2)
parser.add_argument('--test_model', type=str, default='...Pretrained_model root...')
args = parser.parse_args()

args.result_dir = os.path.join('results/', args.save_file_name, 'Best_image/')
#args.test_model = os.path.join('results/', args.save_file_name, 'train_results/', 'Best.pt')

model = build_net()

if torch.cuda.is_available():
    model.cuda()

state_dict = torch.load(args.test_model)
model.load_state_dict(state_dict['model'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
adder = Adder()
model.eval()

datasets = ['100H']

psnr_adder = Adder()
print_network_parameter(model)

for dataset in datasets:
    if not os.path.exists(args.result_dir + dataset + '/'):
        os.makedirs(args.result_dir + dataset)
    print(args.result_dir + dataset)
    dataloader = test_dataloader(os.path.join(args.data_dir), batch_size=1, num_workers=0)
    #factor = 8
    with torch.no_grad():
        psnr_adder = Adder()
        ssim_adder = Adder()

        # Main Evaluation
        for iter_idx, data in enumerate(tqdm(dataloader), 0):
            input_img, label_img, name = data

            input_img = input_img.to(device)

            tm = time.time()
            out3, out2, out1 = model(input_img)

            end = time.time()
            elapsed = end-tm

            adder(elapsed)

            out3_clip = torch.clamp(out3, 0, 1)
            out3_numpy = out3_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            p_numpy = Tensor2_np(out3_numpy)
            label_numpy = Tensor2_np(label_numpy)

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)
            ssim = structural_similarity(p_numpy, label_numpy, data_range=1, channel_axis=2)

            psnr_adder(psnr)
            ssim_adder(ssim)

            if args.save_image:
                save_name = os.path.join(args.result_dir, dataset, name[0])
                out3_clip += 0.5 / 255
                pred = F.to_pil_image(out3_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)

        print('\n runtime %.2f m,\n Average PSNR %.2f dB, Average SSIM %.4f dB'
              % (elapsed, psnr_adder.average(), ssim_adder.average())
              )
        print('==========================================================')
