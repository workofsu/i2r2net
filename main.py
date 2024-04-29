import os
import torch
import argparse
from torch.backends import cudnn
from models.i2r2_model import build_net
from train import _train


def main(args):
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.save_file_name + '/'):
        os.makedirs('results/' + args.save_file_name + '/')
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net()
    print(model)

    if torch.cuda.is_available():
        model.cuda()

    _train(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ###############
    # Directories #
    ###############
    parser.add_argument('--data_dir', type=str, default='D:/Dataset/synthetic/Rain100H/nonpair/Train')
    parser.add_argument('--valid_data', type=str, default='D:/Dataset/synthetic/Rain100H/nonpair/Test')

    #########
    # Train #
    #########
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--test_every', type=int, default=1000, help='# of iter in each epoch')
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=1, help='# of epoch repeated to store model')

    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument("--milestone", type=int, default=[100, 200, 230, 260, 280, 300],
                        help="When to decay learning rate")

    parser.add_argument('--num_worker', type=int, default=2)
    parser.add_argument('--resume', type=str, default='')

    ##############
    # Save image #
    ##############
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])
    parser.add_argument('--save_file_name', default='i2r2_16_100H', type=str)

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', args.save_file_name, 'train_results/')
    args.result_dir = os.path.join('results/', args.save_file_name, 'test')

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    print(args)
    main(args)
