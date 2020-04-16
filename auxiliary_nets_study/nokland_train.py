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
from utils import count_parameters, to_one_hot, dataset_load, allclose_test,\
similarity_matrix, outputs_test, loss_calc, lr_scheduler, optim_init
from settings import parse_args
from models import LocalLossBlockLinear, LocalLossBlockConv, Net, VGGn
import wandb   
    
def train(epoch, lr):
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
        outputs_test(d[0], "outputs/train_tensor_" + str(batch_idx)) 

        y_ = y
        target_onehot = to_one_hot(y, num_classes)
        if args.cuda:
            target_onehot = target_onehot.cuda()

        loss_total = 0
        #h, y = data, target
        y_onehot = target_onehot
        h = d
        for counter in range(len(model.main_cnn.blocks)):
            n = counter
            module =  model.main_cnn.blocks[n]
            auxillery_layer = model.auxillary_nets[n]
            optimizer  = optimizers[n]
            
            if optimizer is not None and not args.backprop:
                optimizer.zero_grad()
            if h is None:
                print(h)


            outputs, h = model(h, n=n)
            if h is None:
                print(h)

            if optimizer is not None and not args.backprop and not isinstance(module, nn.Linear):
                loss = loss_calc(outputs, y, to_one_hot(y), module, args.no_similarity_std)
                loss.backward(retain_graph = False)
                optimizer.step()
                h.detach_()

                loss_total += loss.item()
            if counter == 0:
                print(outputs[1].size())
                outputs_test(outputs[1][0], "outputs/model_tensor_" + str(batch_idx) + "_" + str(counter))
                print(outputs[1][0])
        output = h
        outputs_test(h[0], "outputs/end_tensor_" + str(batch_idx)) 

     
        loss_total_local += loss_total * h.size(0)
        loss = F.cross_entropy(output, y)
        if args.loss_sup == 'predsim' and not args.backprop:
            loss *= (1 - args.beta) 
        loss_total_global += loss.item() * h.size(0)
        if batch_idx <5:
            allclose_test(output[0], epoch, batch_idx)
            print(output[0])
            print()
        else:
            return
        if batch_idx == 4:
            return

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



def test(epoch):
    ''' Run model on test set '''
    model.eval()
    test_loss = 0
    correct = 0
    
    # Loop test set
    batch_idx = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        target_ = target
        target_onehot = to_one_hot(target, num_classes)
        if args.cuda:
            target_onehot = target_onehot.cuda()
        h, y, y_onehot =  data, target, target_onehot
        for counter in range(len(model.main_cnn.blocks)):
           n = counter
           output, h = model(h, n=n)
           if isinstance(model.main_cnn.blocks[n], LocalLossBlockLinear) or isinstance(model.main_cnn.blocks[n], LocalLossBlockConv):
              loss = loss_calc(output, y, y_onehot, model.main_cnn.blocks[n], args.no_similarity_std)
        output = h
        if batch_idx <5:
            allclose_test(output[0], epoch, batch_idx)
            print(output[0])
            print()
        else:
            return
        if batch_idx == 4:
            return
        batch_idx += 1



        test_loss += F.cross_entropy(output, target).item() * data.size(0)
           #test_loss += loss_calc(h, output, y, y_onehot, model.main_cnnmo, auxillery_layer).item()

        pred = output.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target_).cpu().sum()
    
    # Format and print debug string
    loss_average = test_loss / len(test_loader.dataset)
    if args.loss_sup == 'predsim' and not args.backprop:
        loss_average *= (1 - args.beta)

    error_percent = 100 - 100.0 * float(correct) / len(test_loader.dataset)

    wandb.log({"Test Loss Global": loss_average, "Error": error_percent})
    
    
    return loss_average, error_percent
    
   
args = parse_args()
wandb.init(config=args, project='dgl-refactored')

if args.cuda:
    cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
        
torch.manual_seed(25)
if args.cuda:
    torch.cuda.manual_seed(25)


kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
input_dim, input_ch, num_classes, train_transform, dataset_train, train_loader, test_loader = dataset_load(kwargs)


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
            args.dropout, args.nonlin, args.no_similarity_std)
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
    optimizers, _ = optim_init(ncnn, model, args.lr)
else:
    optimizers = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.optim == 'amsgrad')
    
print(model)
print('Model {} has {} parameters influenced by global loss'.format(args.model, count_parameters(model)))

''' The main training and testing loop '''
start_epoch = 1 if checkpoint is None else 1 + checkpoint['epoch']
print(args.epochs, start_epoch)
#args.epochs = 1
for epoch in range(0, 2):#(start_epoch, args.epochs + 1):#(0, 2):##(start_epoch, args.epochs + 1):#range(0, 1):#
    # Decide learning rate
    lr =  lr_scheduler(epoch-1)
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
    train(epoch, lr)
    #train_loss,train_error = train(epoch, lr)

    ##return
    #test_loss,test_error,test_print = test(epoch)

    ## Check if to save checkpoint
    #if args.save_dir is not '':
    #    # Resolve log folder and checkpoint file name
    #    filename = 'chkp_ep{}_lr{:.2e}_trainloss{:.2f}_testloss{:.2f}_trainerr{:.2f}_testerr{:.2f}.tar'.format(
    #            epoch, lr, train_loss, test_loss, train_error, test_error)
    #    dirname = os.path.join(args.save_dir, args.dataset)
    #    dirname = os.path.join(dirname, '{}_mult{:.1f}'.format(args.model, args.feat_mult))
    #    dirname = os.path.join(dirname, '{}_{}x{}_{}_dimdec{}_beta{}_bs{}_drop{}_{}_wd{}_bp{}_lr{:.2e}'.format(
    #            args.nonlin, args.num_layers, args.num_hidden, args.loss_sup + args.loss_sup,  args.dim_in_decoder, args.beta, args.batch_size, args.dropout, args.optim, args.weight_decay, int(args.backprop), args.lr))
    #    
    #    # Create log directory
    #    if not os.path.exists(dirname):
    #        os.makedirs(dirname)
    #    elif epoch==1 and os.path.exists(dirname):
    #        # Delete old files
    #        for f in os.listdir(dirname):
    #            os.remove(os.path.join(dirname, f))
    #    
    #    # Add log entry to log file
    #    with open(os.path.join(dirname, 'log.txt'), 'a') as f:
    #        if epoch == 1:
    #            f.write('{}\n\n'.format(args))
    #            f.write('{}\n\n'.format(model))
    #            if not args.backprop:
    #               f.write('{}\n\n'.format(optimizers[-1]))
    #            else:
    #               f.write('{}\n\n'.format(optimizers))

    #            f.write('Model {} has {} parameters influenced by global loss\n\n'.format(args.model, count_parameters(model)))
    #        f.write(train_print)
    #        f.write(test_print)
    #        f.write('\n')
    #        f.close()
    #    
    #    # Save checkpoint for every epoch
    #    torch.save({
    #        'epoch': epoch,
    #        'args': args,
    #        'state_dict': model.state_dict() if (save_state_dict or epoch==args.epochs) else None,
    #        'train_loss': train_error,
    #        'train_error': train_error,
    #        'test_loss': test_loss,
    #        'test_error': test_error,
    #    }, os.path.join(dirname, filename))  
    #
    #    # Save checkpoint for last epoch with state_dict (for resuming)
    #    torch.save({
    #        'epoch': epoch,
    #        'args': args,
    #        'state_dict': model.state_dict(),
    #        'train_loss': train_error,
    #        'train_error': train_error,
    #        'test_loss': test_loss,
    #        'test_error': test_error,
    #    }, os.path.join(dirname, 'chkp_last_epoch.tar')) 
   
   
