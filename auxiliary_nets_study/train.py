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
AverageMeter, accuracy, lr_scheduler, loss_calc, optim_init, validate
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
    input_dim, input_ch, num_classes, train_transform, dataset_train,\
    train_loader, test_loader = dataset_load(args.dataset, args.batch_size, kwargs)



    if args.model == 'mlp':
        model = Net(args.num_layers, args.num_hidden, input_dim, input_ch, num_classes)
    elif args.model.startswith('vgg'):
        model = VGGn(args.model, input_dim, input_ch, num_classes, args.feat_mult,
            args.dropout, args.nonlin, args.no_similarity_std, args.backprop,
            args.loss_sup, args.dim_in_decoder, args.num_layers,
            args.num_hidden)
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
    if not args.backprop:
        layer_optim, layer_lr = optim_init(ncnn, model, args.lr, args.weight_decay, args.optim)
    else:
        raise NotImplementedError

######################### Lets do the training
    args.epoch = 1
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
            layer_lr[n] = lr_scheduler(args.lr, args.lr_decay_fact, args.lr_decay_milestones, epoch-1)
            optimizer = layer_optim[n]
            if optimizer is not None: 
                for param_group in optimizer.param_groups:
                    param_group['lr'] = layer_lr[n]
        end = time.time()

        for i, (inputs, targets) in enumerate(train_loader):
            # measure data loading time
            #data_time.update(time.time() - end)


            if args.cuda:
                targets = targets.cuda(non_blocking = True)
                inputs = inputs.cuda(non_blocking = True)

            outputs_test(inputs[0], "outputs/train_tensor_" + str(i) + "_" + str(epoch)) 
            target_onehot = to_one_hot(targets)
            target_onehot = target_onehot.cuda()
            #inputs = torch.autograd.Variable(inputs)
            #targets = torch.autograd.Variable(targets)


            #Main loop
            representation = inputs
            end_all = time.time()
            for n in range(ncnn):
                end = time.time()
                optimizer = layer_optim[n]
                #print(model.main_cnn.blocks[n])
                # Forward
                if optimizer is not None:
                    optimizer.zero_grad()

                outputs, representation = model(representation, n=n)
                if optimizer is not None:
                    if n == ncnn-1:
                        print("FDDD")
                        print(representation[0])

                        outputs = representation
                        loss = loss_calc(outputs, targets, target_onehot,
                            model.main_cnn.blocks[n], args.loss_sup, args.beta,
                            args.no_similarity_std)

                        outputs_test(outputs[0], "outputs/end_tensor_" + str(i) + "_" + str(n) + "_" + str(epoch))
                    else:
                        loss = loss_calc(outputs, targets, target_onehot,
                            model.main_cnn.blocks[n], args.loss_sup, args.beta,
                            args.no_similarity_std)
                        print(outputs[1][0]) 
                        outputs_test(outputs[1][0], "outputs/model_tensor_" + str(i) + "_" + str(n) + "_" + str(epoch))
                    wandb.log({"Local Layer " + str(n)+ " Loss": loss.item()})
                    print(loss.item())
                    loss.backward()
                    optimizer.step()  
                    representation.detach_()
                # measure accuracy and record loss
                # measure elapsed time
                batch_time[n].update(time.time() - end)
                #if layer_optim[n] is not None:
                #    if isinstance(model.main_cnn.blocks[n], nn.Linear):
                #        outputs = representation
                #    else:
                #        outputs = outputs[1]
                #    prec1 = accuracy(outputs.data, targets)
                #    losses[n].update(float(loss.item()), float(inputs.size(0)))
                #    top1[n].update(float(prec1[0]), float(inputs.size(0)))
                if n == 0:
                    print(type(outputs))
                    print(outputs[1][0])
                    #outputs_test(outputs[1][0], "outputs/model_tensor_" + str(i) + "_" + str(n))
                    #print(outputs[1][0])
            #print(representation[0])
            print(representation[0])
            #outputs_test(representation[0], "outputs/end_tensor_" + str(i)) 

        for n in range(ncnn):
            ##### evaluate on validation set
            if layer_optim[n] is not None:
                top1test = validate(val_loader, model, epoch, n, loss_sup)
                with open(name_log_txt, "a") as text_file:
                    print("n: {}, epoch {}, loss: {:.5f}, train top1:{} test top1:{} "
                          .format(n+1, epoch, losses[n].avg, top1[n].avg,top1test), file=text_file)


if __name__ == '__main__':
    main()
