from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
from torchvision.datasets.cifar import CIFAR10
from random import randint
import datetime
import itertools
import time
from models import auxillary_classifier2, DGL_Net, VGGn
from settings import parse_args
from utils import to_one_hot, similarity_matrix, dataset_load, outputs_test,\
AverageMeter, accuracy, lr_scheduler, loss_calc, optim_init
import wandb
import numpy as np
np.random.seed(25)
import random
random.seed(25)


##################### Logs
def main():
    global args, best_prec1
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    wandb.init(config=args, project="dgl-refactored")
    if args.cuda:
        cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
            
    torch.manual_seed(25)
    if args.cuda:
        torch.cuda.manual_seed(25)

    time_stamp = str(datetime.datetime.now().isoformat())
    name_log_txt = time_stamp + str(randint(0, 1000)) + args.name
    name_log_txt=name_log_txt +'.log'
    
    with open(name_log_txt, "a") as text_file:
        print(args, file=text_file)
    
    kwargs={}
    input_dim, input_ch, num_classes, train_transform, dataset_train, train_loader, val_loader = dataset_load(kwargs)


    if args.model == 'mlp':
        model = Net(args.num_layers, args.num_hidden, input_dim, input_ch, num_classes)
    elif args.model.startswith('vgg'):
        model = VGGn(args.model, input_dim, input_ch, num_classes, args.feat_mult)
    elif args.model == 'dgl':
        DGL_Net(aux_type=args.type_aux, block_size=args.block_size)
    else:
        print('No valid model defined')

    if args.cuda:
        model = model.cuda()
    wandb.watch(model)
    print(model)
    
    ncnn = len(model.main_cnn.blocks)
    n_cnn = len(model.main_cnn.blocks)
    with open(name_log_txt, "a") as text_file:
        print(model, file=text_file)

    ############### Initialize all
    layer_optim, layer_lr = optim_init(ncnn, model, args.lr)

######################### Lets do the training
    for epoch in range(0, args.epochs+1):
        # Make sure we set the bn right
        model.train()

        #For each epoch let's store each layer individually
        batch_time = [AverageMeter() for _ in range(n_cnn)]
        batch_time_total = AverageMeter()
        data_time = AverageMeter()
        losses = [AverageMeter() for _ in range(n_cnn)]
        top1 = [AverageMeter() for _ in range(n_cnn)]


        for n in range(ncnn):
            layer_lr[n] = lr_scheduler(epoch-1)
            print(layer_lr[n])
            optimizer = layer_optim[n]
            if optimizer is not None: 
                for param_group in optimizer.param_groups:
                    param_group['lr'] = layer_lr[n]
        end = time.time()

        for i, (inputs, targets) in enumerate(train_loader):
            # measure data loading time
            #data_time.update(time.time() - end)
            print(inputs.size(), targets.size())
            outputs_test(inputs[0], "outputs/train_tensor_" + str(i)) 


            if args.cuda:
                targets = targets.cuda(non_blocking = True)
                inputs = inputs.cuda(non_blocking = True)
            #print(inputs[0])
            #inputs = torch.autograd.Variable(inputs)
            #targets = torch.autograd.Variable(targets)


            #Main loop
            representation = inputs
            end_all = time.time()
            for n in range(ncnn):
                end = time.time()
                # Forward
                if layer_optim[n] is not None:
                    layer_optim[n].zero_grad()

                outputs, representation = model(representation, n=n)
                if layer_optim[n] is not None:
                    loss = loss_calc(outputs, targets, to_one_hot(targets), model.main_cnn.blocks[n])
                    loss.backward()
                    layer_optim[n].step()  

                representation = representation.detach()
                # measure accuracy and record loss
                # measure elapsed time
                #batch_time[n].update(time.time() - end)
                #if layer_optim[n] is not None:
                #    if isinstance(model.main_cnn.blocks[n], nn.Linear):
                #        outputs = representation
                #    else:
                #        outputs = outputs[1]
                #    prec1 = accuracy(outputs.data, targets)
                #    losses[n].update(float(loss.item()), float(inputs.size(0)))
                #    top1[n].update(float(prec1[0]), float(inputs.size(0)))
                #    layer_optim[n].zero_grad()
                if n == 0:
                    print(type(outputs))
                    print(outputs[1][0])
                    outputs_test(outputs[1][0], "outputs/model_tensor_" + str(i) + "_" + str(n))
                    #print(outputs[1][0])
            #print(representation[0])
            print(representation[0])
            outputs_test(representation[0], "outputs/end_tensor_" + str(i)) 

        for n in range(ncnn):
            ##### evaluate on validation set
            if layer_optim[n] is not None:
                top1test = validate(val_loader, model, epoch, n)
                with open(name_log_txt, "a") as text_file:
                    print("n: {}, epoch {}, loss: {:.5f}, train top1:{} test top1:{} "
                          .format(n+1, epoch, losses[n].avg, top1[n].avg,top1test), file=text_file)

def validate(val_loader, model, epoch, n):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    all_targs = []
    model.eval()

    end = time.time()
    with torch.no_grad():
        total = 0
        for i, (input, target) in enumerate(val_loader):
            if args.cuda:
                target = target.cuda(non_blocking=True)
                input = input.cuda(non_blocking=True)
            input = torch.autograd.Variable(input)
            target = torch.autograd.Variable(target)

            representation = input
            #output, _ = model(representation, n=n, upto=True)
            for i in range(n):
                output, representation = model(representation, n=i)
                representation = representation.detach()
                # measure accuracy and record loss
                # measure elapsed time 
            output, representation = model(representation, n=n)
            if args.loss_sup == "predsim":
               output = output[1]
            if isinstance(model.main_cnn.blocks[n], nn.Linear):
               output = representation

            loss = F.cross_entropy(output, target)
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)
            losses.update(float(loss.item()), float(input.size(0)))
            top1.update(float(prec1[0]), float(input.size(0)))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            total += input.size(0)

        print(' * Prec@1 {top1.avg:.3f}'
              .format(top1=top1))
        wandb.log({"top1": top1.avg})


    return top1.avg


if __name__ == '__main__':
    main()
