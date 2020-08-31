import argparse
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data
import time
from bisect import bisect_right
import itertools
from models import VGGn
from settings import parse_args
from utils import to_one_hot,  AverageMeter,  loss_calc, test, validate
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
#import wandb
import numpy as np
np.random.seed(25)
import random
random.seed(25)
import sys
#import ast
import random
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
#import gspread
#from oauth2client.service_account import ServiceAccountCredentials


# Training settings
# dgl arguments
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=250, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=400, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=25, metavar='S',
                    help='random seed (default: 1)')
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
parser.add_argument('--n-mlp', type=int, default=0,
                    help='number of hidden fully-connected layers for mlp and vgg models (default: 1')
parser.add_argument('--lr-decay-epoch', type=int, default=80,
                    help='epoch to decay sgd learning rate (default: 80)')
parser.add_argument('--n-conv',  default=0,type=int,
                    help='number of conv layers in aux classifiers')
parser.add_argument('--lr-schd', default='nokland',
                    help='nokland, step, or constant (default: nokland)')
parser.add_argument('--base-lr', type=float, default=1e-4, help='block size')
parser.add_argument('--lr-schedule', nargs='+', type=float, default=[1e-2, 1e-3, 5e-4, 1e-4])
parser.add_argument('--pooling', default="avg", help='pooling type')
parser.add_argument('--bn', action='store_true', default=False,
                    help='batch norm in main model')
parser.add_argument('--aux-bn', action='store_true', default=False,
                    help='batch norm in auxillary layers')
parser.add_argument('--notes', nargs='+', default="none", type=str, help="notes for wandb")

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
            
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
                        
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




def validate(val_loader, model, epoch, n, loss_sup, iscuda):
            #wandb.log({"Layer " + str(n) + " test loss": losses.avg}, step=epoch)
        #wandb.log({"Layer " + str(n) + " top1": top1.avg}, step=epoch)
    return top1.avg

def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    #run = wandb.init(config=args, project="dgl-refactored", notes=args.notes)
    import uuid
    filename = "logs/" + str(uuid.uuid4())
    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    print(filename)
    print(sha)
    print(filename)


    #global args, best_prec1



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

    #url = 'https://app.wandb.ai/muawizc/dgl-refactored/runs/' + run.id
    #insert_row = [sha, args.lr_schd, '', run.id, url, '0', url + "/overview", '', run.notes]


    # data loader
    
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data/CIFAR10', train=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
                         ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=2)


    # Model
    if args.model.startswith('vgg'):
        model = VGGn(args.model, feat_mult=args.feat_mult, dropout=args.dropout,nonlin=args.nonlin,
                      loss_sup= args.loss_sup, dim_in_decoder=args.dim_in_decoder, num_layers=args.num_layers,
            num_hidden = args.num_hidden, aux_type=args.aux_type,
            n_mlp=args.n_mlp, n_conv=args.n_conv, pooling=args.pooling,
            bn=args.bn, aux_bn=args.aux_bn)
    elif args.model == 'resnet18':
        model = resnet18(n_conv=args.n_conv, mlp=args.n_mlp,
                                       block_size=args.block_size,
                                       pooling=args.pooling,
                                       loss_sup=args.loss_sup)
    elif args.model == 'resnet34':
        model = resnet34(n_conv=args.n_conv, mlp=args.n_mlp,
                                       block_size=args.block_size,
                                       pooling=args.pooling,
                                       loss_sup=args.loss_sup)
    elif args.model == 'resnet50':
        model = resnet50(n_conv=args.n_conv, mlp=args.n_mlp,
                                       block_size=args.block_size,
                                       pooling=args.pooling,
                                       loss_sup=args.loss_sup)
    elif args.model == 'resnet101':
        model = resnet101(n_conv=args.n_conv, mlp=args.n_mlp,
                                       block_size=args.block_size,
                                       pooling=args.pooling,
                                       loss_sup=args.loss_sup)
    elif args.model == 'resnet152':
        model = resnet152(n_conv=args.n_conv, mlp=args.n_mlp,
                                       block_size=args.block_size,
                                       pooling=args.pooling,
                                       loss_sup=args.loss_sup)
    else:
        print('No valid model defined')

    #wandb.watch(model)
    #if args.cuda:
    #    model = model.cuda()
    print(model)

    import copy
    model2 = copy.deepcopy(model)
    model.load_state_dict(torch.load('model_0'))
    model2.load_state_dict(torch.load('model_1'))

    n_cnn = len(model.main_cnn.blocks)
    n = n_cnn

######################### Lets do the training
    losses = AverageMeter()
    top1 = AverageMeter()

    model = model.cuda()
    model2 = model2.cuda()

    model.eval()
    model2.eval()
    with torch.no_grad():
        total = 0
        histogram = []
        for i in range(256):
            histogram.append(AverageMeter())
        for i, (input, target) in enumerate(val_loader):
            print(i)
            target = target#.cuda(non_blocking=True)
            input = input#.cuda(non_blocking=True)

            representation = input
            array = [representation]
            array_2 = []
            for k in range(n_cnn-1):
                for rep in array:
                    rep_ = rep.cuda()
                    #rep = rep.cuda(non_blocking=True)
                    output, _, rep_1 = model(rep_, n=k)
                    array_2.append(rep_1.cpu())
                    output, _, rep_2 = model2(rep_, n=k)
                    array_2.append(rep_2.cpu())
                    #del rep
                array = array_2
                array_2 = []
            for rep in array:
                #rep = rep.cuda(non_blocking=True)
                output, _, rep_1 = model(rep, n=n-1)
                array_2.append(output)
                output, _, rep_2 = model2(rep, n=n-1)
                array_2.append(output)
            for c, out in enumerate(array_2):
               histogram[c].update(float(accuracy(out.data, target)[0]), float(input.size(0)))
               print(histogram[c].avg, str(float(accuracy(out.data, target)[0])))
            array = torch.mean(torch.stack(array_2, dim=0), dim=0)
            # measure accuracy and record loss
            loss = F.cross_entropy(array, target)
            losses.update(float(loss.item()), float(input.size(0)))
            prec1 = accuracy(array.data, target)
            print(str(float(prec1[0])), str(float(loss.item())))
            top1.update(float(prec1[0]), float(input.size(0)))

            total += input.size(0)

        hist = []
        for a in histogram:
            print(a.avg)
            hist.append(a.avg)
        
        fig = plt.hist(hist, bins=10)
        plt.savefig('histogram.png')

        print()
        print(top1.avg, losses.avg) 



if __name__ == '__main__':
    main()
