import argparse
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import time
from bisect import bisect_right
import itertools
from models import DGL_Net, VGGn
from settings import parse_args
from utils import to_one_hot,  AverageMeter,  loss_calc, test, validate
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
#import wandb
import numpy as np
np.random.seed(25)
import random
random.seed(25)
import sys

import torch.optim as optim
from torchvision import datasets, transforms

import uuid
filename = str(uuid.uuid4())
import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
print(filename)
sys.stdout = open(filename, "w",buffering=1)
print(sha)
print(" ".join(str(item) for item in sys.argv[0:]))

# Training settings
# dgl arguments
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=400, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=25, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--type_aux', type=str, default='mlp', metavar='N')
parser.add_argument('--block_size', type=int, default=1, help='block size')
parser.add_argument('--name', default='', type=str, help='name')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--lr', type=float, default=5e-4, help='block size')

# localloss arguments
parser.add_argument('--model', default='vgg8b',
                    help='model, mlp, vgg13, vgg16, vgg19, vgg8b, vgg11b (default: vgg8b)')
parser.add_argument('--num-layers', type=int, default=1,
                    help='number of hidden fully-connected layers for mlp and vgg models (default: 1')
parser.add_argument('--num-hidden', type=int, default=1024,
                    help='number of hidden units for mpl model (default: 1024)')
parser.add_argument('--dim-in-decoder', type=int, default=4096,
                    help='input dimension of decoder_y used in pred and predsim loss (default: 4096)')
parser.add_argument('--feat-mult', type=float, default=1,
                    help='multiply number of CNN features with this number (default: 1)')
parser.add_argument('--lr-decay-milestones', nargs='+', type=int, default=[200, 300, 350, 375],
                    help='decay learning rate at these milestone epochs (default: [200,300,350,375])')
parser.add_argument('--lr-decay-fact', type=float, default=0.25,
                    help='learning rate decay factor to use at milestone epochs (default: 0.25)')
parser.add_argument('--optim', default='adam',
                    help='optimizer, adam, amsgrad or sgd (default: adam)')
parser.add_argument('--momentum', type=float, default=0.0,
                    help='SGD momentum (default: 0.0)')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='weight decay (default: 0.0)')
parser.add_argument('--beta', type=float, default=0.99,
                    help='fraction of similarity matching loss in predsim loss (default: 0.99)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout after each nonlinearity (default: 0.0)')
parser.add_argument('--loss-sup', default='predsim',
                    help='supervised local loss, sim or pred (default: predsim)')
parser.add_argument('--nonlin', default='relu',
                    help='nonlinearity, relu or leakyrelu (default: relu)')
parser.add_argument('--no-similarity-std', action='store_true', default=False,
                    help='disable use of standard deviation in similarity matrix for feature maps')
parser.add_argument('--aux-type', default='nokland',
                    help='nonlinearity, relu or leakyrelu (default: relu)')
parser.add_argument('--mlp-layers', type=int, default=0,
                    help='number of hidden fully-connected layers for mlp and vgg models (default: 1')
parser.add_argument('--lr-decay-epoch', type=int, default=80,
                    help='epoch to decay sgd learning rate (default: 80)')
parser.add_argument('--nlin',  default=3,type=int,
                    help='number of conv layers in aux classifiers')




args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

##################### Logs
def lr_scheduler(lr, epoch, args):
    if args.optim == "adam":
        lr = lr * args.lr_decay_fact ** bisect_right(args.lr_decay_milestones, (epoch))
    elif args.optim == "sgd":
        if (epoch+2) % args.lr_decay_epoch == 0:
            lr = lr * args.lr_decay_fact
        else:
            lr = lr
    return lr

def optim_init(ncnn, model, lr, weight_decay, optimizer):
    layer_optim = [None] * ncnn
    layer_lr = [lr] * ncnn
    for n in range(ncnn):
        to_train = itertools.chain(model.main_cnn.blocks[n].parameters(), model.auxillary_nets[n].parameters())
        if len(list(to_train)) != 0:
            to_train = itertools.chain(model.main_cnn.blocks[n].parameters(), model.auxillary_nets[n].parameters())
            if args.optim == "adam":
                layer_optim[n] = optim.Adam(to_train, lr=layer_lr[n], weight_decay=weight_decay, amsgrad=optimizer == 'amsgrad')
            elif args.optim == "sgd":
                layer_optim[n] = optim.SGD(to_train, lr=layer_lr[n], momentum=0.9, weight_decay=weight_decay)
        else:
            layer_optim[n] = None
    return layer_optim, layer_lr


def main():
    global args, best_prec1
    #wandb.init(config=args, project="dgl-refactored")

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    print(sha)
    if args.cuda:
        cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
            
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # data loader
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
    ])
    dataset_train = datasets.CIFAR10('../data/CIFAR10', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=None,
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data/CIFAR10', train=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
                         ])),
        batch_size=args.batch_size, shuffle=False)


    # Model
    if args.model.startswith('vgg'):
        model = VGGn(args.model, feat_mult=args.feat_mult, dropout=args.dropout,nonlin=args.nonlin, no_similarity_std=args.no_similarity_std,
                      loss_sup= args.loss_sup, dim_in_decoder=args.dim_in_decoder, num_layers=args.num_layers,
            num_hidden = args.num_hidden, aux_type=args.aux_type,
            mlp_layers=args.mlp_layers, nlin=args.nlin)
    elif args.model == 'dgl':
        DGL_Net(aux_type=args.type_aux, block_size=args.block_size)
    elif args.model == 'resnet18':
        model = resnet18(nlin=args.nlin, mlp=args.mlp_layers,
                                       block_size=args.block_size)
    elif args.model == 'resnet34':
        model = resnet34(nlin=args.nlin, mlp=args.mlp_layers,
                                       block_size=args.block_size)
    elif args.model == 'resnet50':
        model = resnet50(nlin=args.nlin, mlp=args.mlp_layers,
                                       block_size=args.block_size)
    elif args.model == 'resnet101':
        model = resnet101(nlin=args.nlin, mlp=args.mlp_layers,
                                       block_size=args.block_size)
    elif args.model == 'resnet152':
        model = resnet152(nlin=args.nlin, mlp=args.mlp_layers,
                                       block_size=args.block_size)
    else:
        print('No valid model defined')

    if args.cuda:
        model = model.cuda()
    print(model)

    n_cnn = len(model.main_cnn.blocks)


    # Define optimizer en local lr
    layer_optim, layer_lr = optim_init(n_cnn, model, args.lr, args.weight_decay, args.optim)

######################### Lets do the training
    for epoch in range(0, args.epochs+1):
        # Make sure we set the bn right
        model.train()
        top1 = [AverageMeter() for _ in range(n_cnn)]


        for n in range(n_cnn):
            layer_lr[n] = lr_scheduler(layer_lr[n], epoch-1, args)
            optimizer = layer_optim[n]
            if optimizer is not None: 
                for param_group in optimizer.param_groups:
                    param_group['lr'] = layer_lr[n]

        for i, (inputs, targets) in enumerate(train_loader):
            if args.cuda:
                targets = targets.cuda(non_blocking = True)
                inputs = inputs.cuda(non_blocking = True)

            target_onehot = to_one_hot(targets)
            if args.cuda:
                target_onehot = target_onehot.cuda()


            representation = inputs
            for n in range(n_cnn):
                optimizer = layer_optim[n]

                # Forward
                if optimizer is not None: # some layers are just down-samplings for instance
                    optimizer.zero_grad()

                outputs, representation = model(representation, n=n)

                if optimizer is not None:
                    if isinstance(model.main_cnn.blocks[n], torch.nn.Linear): # if final layer, the representation is empty
                        outputs = representation
                    loss = loss_calc(outputs, targets, target_onehot,
                            model.main_cnn.blocks[n], args.loss_sup, args.beta,
                            args.no_similarity_std)

                    loss.backward()
                    optimizer.step()  
                    representation.detach_()


        # We now log the statistics
        print('epoch: ' + str(epoch) + ' , lr: ' + str(lr_scheduler(layer_lr[-1], epoch-1, args)))
        test(epoch, model, test_loader)
        for n in range(n_cnn):
            if layer_optim[n] is not None:
                top1test = validate(test_loader, model, epoch, n, args.loss_sup, args.cuda)
                print("n: {}, epoch {}, test top1:{} "
                      .format(n + 1, epoch, top1test))





if __name__ == '__main__':
    main()
