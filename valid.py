import torch
from torchvision.transforms import functional as F
from data import test_dataloader
from utils import Adder, Tensor2_np
import os
import time
import numpy
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import torch.nn.functional as f

def _valid(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    valid_data = test_dataloader(args.valid_data, batch_size=1, num_workers=args.num_worker)
    model.eval()
    time_adder = Adder()
    psnr_adder = Adder()
    ssim_adder = Adder()

    with torch.no_grad():
        print('-' * 100)
        print('Start Derain Evaluation')
        for idx, data in enumerate(valid_data):
            input_img, label_img, name = data

            input_img = input_img.to(device)

            tm = time.time()
            out3, _, _ = model(input_img)
            elapsed = time.time() - tm
            time_adder(elapsed)

            out3_clip = torch.clamp(out3, 0, 1)   #####################out 바꾸기#####################################
            out3_numpy = out3_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            p_numpy = Tensor2_np(out3_numpy)
            label_numpy = Tensor2_np(label_numpy)

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)
            ssim = structural_similarity(p_numpy, label_numpy, data_range=1, channel_axis=2)

            if not os.path.exists(os.path.join(args.result_dir)):
                os.mkdir(os.path.join(args.result_dir))

            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                out3_clip += 0.5 / 255
                pred = F.to_pil_image(out3_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)

            psnr_adder(psnr)
            ssim_adder(ssim)
            print('\r%03d' % (idx+1), end=' ')
    model.train()
    return time_adder.total(), psnr_adder.average(), ssim_adder.average()
