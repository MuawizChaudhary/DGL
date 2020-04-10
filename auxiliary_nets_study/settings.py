import argparse
import torch

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=25, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--type_aux', type=str, default='mlp',metavar='N')
    parser.add_argument('--block_size', type=int, default=1, help='block size')
    parser.add_argument('--name', default='',type=str,help='name')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--lr', type=float, default=5e-4, help='block size')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='number of hidden fully-connected layers for mlp and vgg models (default: 1')
    parser.add_argument('--num-hidden', type=int, default=1024,
                        help='number of hidden units for mpl model (default: 1024)')
    parser.add_argument('--dim-in-decoder', type=int, default=4096,
                        help='input dimension of decoder_y used in pred and predsim loss (default: 4096)')
    parser.add_argument('--no-print-stats', action='store_true', default=False,
                        help='do not print layerwise statistics during training with local loss')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout after each nonlinearity (default: 0.0)')
    parser.add_argument('--loss-sup', default='predsim',
                        help='supervised local loss, sim or pred (default: predsim)')
    parser.add_argument('--nonlin', default='relu',
                        help='nonlinearity, relu or leakyrelu (default: relu)')
    parser.add_argument('--no-similarity-std', action='store_true', default=False,
                        help='disable use of standard deviation in similarity matrix for feature maps')
    parser.add_argument('--backprop', action='store_true', default=False,
                        help='disable local loss training')


    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

