import argparse
import numpy as np
np.random.seed(25)
import torch
torch.manual_seed(25)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
random.seed(25)
from torch.backends import cudnn
from bisect import bisect_right
import math
import os
import itertools
from utils import count_parameters, to_one_hot, dataset_load,\
similarity_matrix,  loss_calc, lr_scheduler, optim_init, test
from settings import parse_args
from models import LocalLossBlockLinear, LocalLossBlockConv, Net, VGGn
import wandb





def train(epoch, lr, ncnn):
    ''' Train model on train set'''
    model.train()
    correct = 0
    loss_total_local = 0
    loss_total_global = 0
    
    # Add progress bar
    if args.progress_bar:
        pbar = tqdm(total=len(train_loader))
        
    
    # Loop train set
    for batch_idx, (d, y) in enumerate(train_loader):
        if args.cuda:
            d, y = d.cuda(), y.cuda()
        #print(d.size())
        #outputs_test(d[0], "outputs/train_tensor_" + str(batch_idx) + "_" + str(epoch))

        y_ = y
        target_onehot = to_one_hot(y, num_classes)
        if args.cuda:
            target_onehot = target_onehot.cuda()

        loss_total = 0
        #h, y = data, target
        y_onehot = target_onehot
        h = d
        for counter in range(ncnn):
            n = counter
            optimizer  = optimizers[n]
            
            if optimizer is not None and not args.backprop:
                optimizer.zero_grad()
            
            outputs, h = model(h, n=n)

            if optimizer is not None and not args.backprop and not isinstance(model.main_cnn.blocks[n], nn.Linear):
                loss = loss_calc(outputs, y, y_onehot, model.main_cnn.blocks[n],
                        args.loss_sup, args.beta, args.no_similarity_std)
                loss.backward(retain_graph = False)
                optimizer.step()
                h.detach_()
                loss_total += loss.item()
        output = h
        loss_total_local += loss_total * h.size(0)
        loss = F.cross_entropy(output, y)
        if args.loss_sup == 'predsim' and not args.backprop:
            loss *= (1 - args.beta) 
        loss_total_global += loss.item() * h.size(0)

        # Backward pass and optimizer step
        # For local loss functions, this will only affect output layer
        loss.backward()
        if not args.backprop:
            classifier_optim = optimizers[-1]
        else: 
            classifier_optim = optimizers
        
        classifier_optim.step()
        classifier_optim.zero_grad()
        pred = output.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(y_).cpu().sum()
        
        # Update progress bar
        if args.progress_bar:
            pbar.set_postfix(loss=loss.item(), refresh=False)
            pbar.update()
            
    if args.progress_bar:
        pbar.close()
        
    # Format and print debug string
    loss_average_local = loss_total_local / len(train_loader.dataset)
    loss_average_global = loss_total_global / len(train_loader.dataset)
    error_percent = 100 - 100.0 * float(correct) / len(train_loader.dataset)
    
    return loss_average_local+loss_average_global, error_percent




    
   
args = parse_args()
wandb.init(config=args, project='dgl-refactored')


import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
print(sha)






if args.cuda:
    cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
        
torch.manual_seed(25)
if args.cuda:
    torch.cuda.manual_seed(25)


kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
input_dim, input_ch, num_classes, train_transform, dataset_train,\
train_loader,\
test_loader = dataset_load(args.dataset, args.batch_size, kwargs)


checkpoint = None
if not args.resume == '':
    if os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
        args.model = checkpoint['args'].model
        args_backup = args
        args = checkpoint['args']
        args.optim = args_backup.optim
        args.momentum = args_backup.momentum
        args.weight_decay = args_backup.weight_decay
        args.dropout = args_backup.dropout
        print('=> loaded checkpoint "{}" (epoch {})'.format(args.resume, checkpoint['epoch']))
    else:
        print('Checkpoint not found: {}'.format(args.resume))
        
if args.model == 'mlp':
    model = Net(args.num_layers, args.num_hidden, input_dim, input_ch, num_classes)
elif args.model.startswith('vgg'):
    model = VGGn(args.model, input_dim, input_ch, num_classes, args.feat_mult,
            args.dropout, args.nonlin, args.no_similarity_std, args.backprop,
            args.loss_sup, args.dim_in_decoder, args.num_layers,
            args.num_hidden)
else:
    print('No valid model defined')
print(args.model, input_dim, input_ch, num_classes, args.feat_mult)


# Check if to load model
if checkpoint is not None:
    model.load_state_dict(checkpoint['state_dict'])
    args = args_backup
    
if args.cuda:
    model.cuda()
wandb.watch(model)

if args.progress_bar:
    from tqdm import tqdm
ncnn = len(model.main_cnn.blocks)
if not args.backprop:
    optimizers, _ = optim_init(ncnn, model, args.lr, args.weight_decay, args.optim)
else:
    optimizers = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.optim == 'amsgrad')
    
print(model)
print('Model {} has {} parameters influenced by global loss'.format(args.model, count_parameters(model)))

''' The main training and testing loop '''
start_epoch = 1 if checkpoint is None else 1 + checkpoint['epoch']
print(args.epochs, start_epoch)
#args.epochs = 1
for epoch in range(start_epoch, args.epochs + 1):
    # Decide learning rate
    print(args.lr, args.lr_decay_fact, args.lr_decay_milestones, epoch-1)
    lr =  lr_scheduler(args.lr, args.lr_decay_fact, args.lr_decay_milestones, epoch-1)
    print(lr)
    save_state_dict = False
    for ms in args.lr_decay_milestones:
        if (epoch-1) == ms:
            print('Decaying learning rate to {}'.format(lr))
            decay = True
        elif epoch == ms:
            save_state_dict = True

    # Set learning rate
    if not args.backprop:
        for counter, module in enumerate(model.main_cnn.blocks):
            if optimizers[counter] is not None:
                for param_group in optimizers[counter].param_groups:
                    param_group['lr'] = lr
    else:
        for param_group in optimizers.param_groups:
            param_group['lr'] = lr
    
    # Train and test    
    print(epoch, lr)

    train_loss,train_error = train(epoch, lr, ncnn)
    print('epoch: '+str(epoch)+' , lr : '+str(lr))
    test_loss, test_error = test(epoch,model, test_loader)
    for n in range(ncnn):
        ##### evaluate on validation set
        if layer_optim[n] is not None:
            top1test = validate(test_loader, model, epoch, n, args.loss_sup, args.cuda)
            with open(name_log_txt, "a") as text_file:
                print("n: {}, epoch {}, loss: {:.5f}, train top1:{} test top1:{} "
                      .format(n + 1, epoch, losses[n].avg, top1[n].avg, top1test), file=text_file)


