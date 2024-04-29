import os
import torch
from decimal import Decimal
from data import train_dataloader
from utils import Adder, Timer
from torch.utils.tensorboard import SummaryWriter
from importlib import import_module
from valid import _valid
from loss.ssim import SSIM
#ssim = SSIM().cuda()
import torch.nn.functional as F
import torch.nn as nn

from warmup_scheduler import GradualWarmupScheduler


def print_network_parameter(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)


def _train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mse_loss = torch.nn.MSELoss()

    module = import_module('loss.ssim')
    ssim_loss = getattr(module, 'SSIM')().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)

    dataloader = (train_dataloader(
        args.data_dir, args.batch_size, args.patch_size, args.num_worker, use_transform=True, test_every=args.test_every
        )
    )
    max_iter = len(dataloader)
    #warmup_epochs = 3
    #scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch - warmup_epochs,eta_min=1e-6)
    #scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestone, gamma=args.gamma)
    #scheduler.step()
    epoch = 1
    best_psnr= -1
    best_epoch = -1
    best_ssim = -1

    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        optimizer.step()
        scheduler.step(epoch=epoch)

        best_state = torch.load(args.resume.replace('model', 'Best'))
        best_epoch = best_state['epoch']
        best_psnr = best_state['PSNR']
        best_ssim = best_state['SSIM']

        #optimizer.step()

        print('Resume from %d'%epoch)
        epoch += 1

    writer = SummaryWriter(os.path.join('results/', args.save_file_name, 'runs'))

    #epoch_mse_adder = Adder()
    #epoch_ssim_adder = Adder()
    iter_mse_adder = Adder()
    iter_ssim_adder = Adder()

    epoch_timer = Timer('m')
    iter_timer = Timer('m')

    print_network_parameter(model)

    for epoch_idx in range(epoch, args.num_epoch + 1):
        print("[Epoch: %03d]  [LR: %.2e]" % (epoch_idx, Decimal(scheduler.get_last_lr()[0])))

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):

            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()
            out3, out2, out1 = model(input_img)    ###################### model output #########################

            l3_mse = mse_loss(out3, label_img)
            l2_mse = mse_loss(out2, label_img)
            l1_mse = mse_loss(out1, label_img)
            mse_content = l1_mse + l2_mse + l3_mse

            l3_ssim = ssim_loss(out3, label_img)
            l2_ssim = ssim_loss(out2, label_img)
            l1_ssim = ssim_loss(out1, label_img)
            ssim_content = (1-l1_ssim) + (1-l2_ssim) + (1-l3_ssim)

            loss = mse_content + 0.2 * ssim_content
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)

            optimizer.step()

            iter_mse_adder(mse_content.item())
            iter_ssim_adder(ssim_content.item())

            #epoch_mse_adder(mse_content.item())
            #epoch_ssim_adder(ssim_content.item())

            if (iter_idx + 1) % args.print_freq == 0:
                print("[Time: %7.4f] [Iter: %2d/%2d] [MSE: %7.9f] [SSIM: %7.9f]" % (
                    iter_timer.toc(),
                    (iter_idx + 1) * args.batch_size,
                    max_iter * args.batch_size,
                    iter_mse_adder.average(),
                    iter_ssim_adder.average())
                      )
                writer.add_scalar('MSE Loss', iter_mse_adder.average(), iter_idx + (epoch_idx-1) * max_iter)
                writer.add_scalar('1-SSIM Loss', iter_ssim_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                
                iter_timer.tic()
                iter_mse_adder.reset()
                iter_ssim_adder.reset()
        scheduler.step()

        overwrite_name = os.path.join(args.model_save_dir, 'model.pt')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pt' % epoch_idx)
            torch.save({'model': model.state_dict()}, save_name)
        print("Elapsed time: %4.2f" % (
            epoch_timer.toc())
              )

        #scheduler.step()

        if epoch_idx % args.save_freq == 0:
            t_time, val_psnr, val_ssim = _valid(model, args, epoch_idx)
            print('\n[%d epoch] || [Time: %.3f] \n PSNR %.3f dB || SSIM %.4f ((Best: epoch %d | PSNR %.2f | SSIM %.4f))'
                  % (epoch_idx, t_time, val_psnr, val_ssim, best_epoch, best_psnr, best_ssim)
                  )
            print('-' * 100)

            writer.add_scalar('PSNR_DeRain', val_psnr, epoch_idx)
            writer.add_scalar('SSIM_DeRain', val_ssim, epoch_idx)

            if val_psnr >= best_psnr:
                torch.save({'model': model.state_dict(),
                            'epoch': epoch_idx,
                            'PSNR': val_psnr,
                            'SSIM': val_ssim}, os.path.join(args.model_save_dir, 'Best.pt'))
                best_psnr = val_psnr
                best_epoch = epoch_idx
                best_ssim = val_ssim

    save_name = os.path.join(args.model_save_dir, 'Final.pt')
    torch.save({'model': model.state_dict()}, save_name)
