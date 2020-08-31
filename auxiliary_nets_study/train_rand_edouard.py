#! /usr/bin/env python3


import argparse
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import time
from bisect import bisect_right
import itertools
from models import VGGn
from settings import parse_args
from utils import to_one_hot, AverageMeter, loss_calc, test, validate
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
# import wandb
import numpy as np

np.random.seed(25)
import random

random.seed(25)
import sys
# import ast
import random
import torch.optim as optim
from torchvision import datasets, transforms

# import gspread
# from oauth2client.service_account import ServiceAccountCredentials


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
parser.add_argument('--n-conv', default=0, type=int,
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

parser.add_argument('--edouard', action='store_true', default=False,
                    help='display edouard stuff')

parser.add_argument('--edouard2', action='store_true', default=False,
                    help='display edouard stuff')


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





##################### Logs
def lr_scheduler(lr, epoch, args):
    if args.optim == "adam":
        lr = lr * args.lr_decay_fact ** bisect_right(args.lr_decay_milestones, (epoch))
    elif args.optim == "adam" or args.optim == "sgd":
        if (epoch + 2) % args.lr_decay_epoch == 0:
            lr = lr * args.lr_decay_fact
        else:
            lr = lr
    return lr


def optim_init(ncnn, model, args):
    layer_optim = [None] * ncnn
    layer_lr = [args.lr] * ncnn
    for n in range(ncnn):
        to_train = itertools.chain(model.main_cnn.blocks[n].parameters(), model.auxillary_nets[n].parameters())
        if args.optim == "adam":
            layer_optim[n] = optim.Adam(to_train, lr=layer_lr[n],
                                        weight_decay=args.weight_decay, amsgrad=args.optim == 'amsgrad')
        elif args.optim == "sgd":
            layer_optim[n] = optim.SGD(to_train, lr=layer_lr[n],
                                       momentum=args.momentum, weight_decay=args.weight_decay)
    return layer_optim, layer_lr


def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # run = wandb.init(config=args, project="dgl-refactored", notes=args.notes)
    import uuid
    filename = "logs/" + str(uuid.uuid4())
    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    print(filename)
    print(sha)
    print(filename)

    # global args, best_prec1



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

    # url = 'https://app.wandb.ai/muawizc/dgl-refactored/runs/' + run.id
    # insert_row = [sha, args.lr_schd, '', run.id, url, '0', url + "/overview", '', run.notes]


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
        batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data/CIFAR10', train=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
                         ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    # wandb.watch(model)

    N = 2
    import copy

    models = []

    for i in range(0, N ):
        model = None


        # Model
        if args.model.startswith('vgg'):
            model = VGGn(args.model, feat_mult=args.feat_mult, dropout=args.dropout, nonlin=args.nonlin,
                         loss_sup=args.loss_sup, dim_in_decoder=args.dim_in_decoder, num_layers=args.num_layers,
                         num_hidden=args.num_hidden, aux_type=args.aux_type,
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

        if args.cuda:
            model = model.cuda()

        models.append(model)



    n_cnn = len(models[0].main_cnn.blocks)

    # scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

    # creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)

    # client = gspread.authorize(creds)

    # sheet = client.open("Spreadsheet DGL").sheet1
    # sheet.append_row(insert_row, table_range='A1')

    # Define optimizer en local lr
    layer_optim = []
    layer_lr = []
    for i in range(0, N):
        model_optim, model_lr = optim_init(n_cnn, models[i], args)
        layer_optim.append(model_optim)
        layer_lr.append(model_lr)
    ######################### Lets do the training
    for epoch in range(0, args.epochs + 1):
        # Make sure we set the bn right
        for c in range(0, N):
            models[c].train()
            torch.save(models[c].state_dict(), 'model_' + str(c))
        losses = [[AverageMeter() for _ in range(n_cnn)] for _ in range(0, N)]

        for i, (inputs, targets) in enumerate(train_loader):
            if args.cuda:
                targets = targets.cuda(non_blocking=True)
                inputs = inputs.cuda(non_blocking=True)

            target_onehot = to_one_hot(targets, 10)
            if args.cuda:
                target_onehot = target_onehot.cuda(non_blocking=True)

            representation = inputs
            for n in range(n_cnn):
                # Forward
                index = random.randint(0, N - 1)
                optimizer = layer_optim[index][n]
                pred, sim, representation = models[index](representation, n=n)
                loss = loss_calc(pred, sim, targets, target_onehot,
                                 args)  # .loss_sup, args.beta,
                # args.no_similarity_std)
                loss.backward()
                optimizer.step()
                representation.detach_()
                losses[index][n].update(float(loss.item()), float(targets.size(0)))
                optimizer.zero_grad()
        if args.lr_schd == 'nokland' or args.lr_schd == 'step':
            for c in range(N):
                for n in range(n_cnn):
                    layer_lr[c][n] = lr_scheduler(layer_lr[c][n], epoch - 1, args)
                    optimizer = layer_optim[c][n]
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = layer_lr[c][n]
        elif args.lr_schd == 'constant':
            closest_i = max([c for c, i in enumerate(args.lr_decay_milestones) if i <= epoch])
            for c in range(N):
                for n in range(n_cnn):
                    layer_lr[c][n] = lr_scheduler(layer_lr[c][n], epoch - 1, args)
                    optimizer = layer_optim[c][n]
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr_schedule[closest_i]

        # We now log the statistics
        print('epoch: ' + str(epoch) + ' , lr: ' + str(lr_scheduler(layer_lr[0][-1], epoch - 1, args)))
        if args.edouard:
            for c in range(0, N):
                for n in range(n_cnn):
                    if layer_optim[c][n] is not None:
                        #wandb.log({"Layer " + str(n) + " train loss": losses[n].avg}, step=epoch)
                        top1test = validate(test_loader, models[c], epoch, n, args.loss_sup, args.cuda)
                        print("CNN {}- n: {}, epoch {}, test top1:{} ".format(c, n + 1, epoch, top1test))
        if args.edouard2:
            rand_layer = [random.randint(0, N-1) for iter in range(n_cnn)]
            for c in range(0, N):
                models[c].eval()

            with torch.no_grad():
                top1= AverageMeter()

                for i, (input, target) in enumerate(test_loader):
                    target = target.cuda(non_blocking=True)
                    input = input.cuda(non_blocking=True)

                    representation = input
                    pred=[]
                    for k in range(n_cnn):
                        pred, sim, representation = models[rand_layer[k]](representation, n=k)
                    prec1 = accuracy(pred.data, target)
                    print(str(float(prec1[0])), str(float(loss.item())))
                    top1.update(float(prec1[0]), float(input.size(0)))


            print("random test, test top1:{} ".format(top1.avg))


if __name__ == '__main__':
    main()
