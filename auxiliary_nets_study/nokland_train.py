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
from nokland_utils import count_parameters, to_one_hot, dataset_load, allclose_test, similarity_matrix
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
        
    # Clear layerwise statistics
    if not args.no_print_stats:
        for m in model.modules():
            if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                m.clear_stats()
    
    # Loop train set
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        target_ = target
        target_onehot = to_one_hot(target, num_classes)
        if args.cuda:
            target_onehot = target_onehot.cuda()
        
        loss_total = 0

        h, y, y_onehot =  data, target, target_onehot
        for counter in range(len(model.features)):
            n = counter
            module = model.features[n]
            auxillery_layer = model.auxillery_layers[n]
            if not args.backprop:
                optimizer  = optimizers[n]
            if isinstance(module, LocalLossBlockLinear) or isinstance(module, LocalLossBlockConv): 
                if counter == len(model.features) - 2 and isinstance(module, LocalLossBlockLinear):
                    h = h.view(h.size(0), -1)
                h, h_return = module(h)
                if not args.backprop:
                    loss = loss_calc(h, h_return, y, y_onehot, module, auxillery_layer)
                    h = backward_optimize(module, loss, h_return, optimizer)
                    loss = loss.item()
                    loss_total += loss
                else:
                    h = h_return
            elif isinstance(module, nn.MaxPool2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Identity):
                h = module(h)
           
        output = h
     
        loss_total_local += loss_total * output.size(0)
        loss = F.cross_entropy(output, target)
        if args.loss_sup == 'predsim' and not args.backprop:
            loss *= (1 - args.beta) 
        loss_total_global += loss.item() * data.size(0)
        #if batch_idx <5:
        #    allclose_test(output[0], epoch, batch_idx)
        #    print(output[0])
        #    print()
        #else:
        #    return
        #if batch_idx == 4:
        #    return

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
        correct += pred.eq(target_).cpu().sum()
        
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
    string_print = 'Train epoch={}, lr={:.2e}, loss_local={:.4f}, loss_global={:.4f}, error={:.3f}%, mem={:.0f}MiB, max_mem={:.0f}MiB\n'.format(
        epoch,
        lr, 
        loss_average_local,
        loss_average_global,
        error_percent,
        torch.cuda.memory_allocated()/1e6,
        torch.cuda.max_memory_allocated()/1e6)
    if not args.no_print_stats:
        for m in model.modules():
            if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                string_print += m.print_stats() 
    print(string_print)
    
    return loss_average_local+loss_average_global, error_percent, string_print


def loss_calc(h, h_return, y, y_onehot, module, auxillery):
    if args.loss_sup == 'sim' or args.loss_sup == 'predsim':
        h_loss = auxillery.sim_loss(h)
        Rh = similarity_matrix(h_loss)
       
    # Calculate supervised loss
    if args.loss_sup == 'sim':
        Ry = similarity_matrix(y_onehot).detach()
        loss_sup = F.mse_loss(Rh, Ry)
        if not args.no_print_stats:
            self.loss_sim += loss_sup.item() * h.size(0)
            self.examples += h.size(0)
    elif args.loss_sup == 'pred':
        h = auxillery.avg_pool(h)
        y_hat_local = auxillery.decoder_y(h.view(h.size(0), -1))
        loss_sup = F.cross_entropy(y_hat_local,  y.detach())
        if not args.no_print_stats:
            self.loss_pred += loss_sup.item() * h.size(0)
            self.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
            self.examples += h.size(0)
    elif args.loss_sup == 'predsim':                    
        h = auxillery.avg_pool(h)
        y_hat_local = auxillery.decoder_y(h.view(h.size(0), -1))
        Ry = similarity_matrix(y_onehot).detach()
        loss_pred = (1-args.beta) * F.cross_entropy(y_hat_local,  y.detach())
        loss_sim = args.beta * F.mse_loss(Rh, Ry)
        loss_sup = loss_pred + loss_sim
        if not args.no_print_stats:
            module.loss_pred += loss_pred.item() * h.size(0)
            module.loss_sim += loss_sim.item() * h.size(0)
            module.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
            module.examples += h.size(0)
    return loss_sup


def backward_optimize(model, loss, h_return, optimizer):
    if model.training:
        loss.backward(retain_graph = False)
        optimizer.step()
        optimizer.zero_grad()
        h_return.detach_()
    return h_return


def test(epoch):
    ''' Run model on test set '''
    model.eval()
    test_loss = 0
    correct = 0
    
    # Clear layerwise statistics
    if not args.no_print_stats:
        for m in model.modules():
            if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                m.clear_stats()
    
    # Loop test set
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        target_ = target
        target_onehot = to_one_hot(target, num_classes)
        if args.cuda:
            target_onehot = target_onehot.cuda()
        
        h, y, y_onehot =  data, target, target_onehot
        with torch.no_grad():
            for counter in range(len(model.features)):
                n = counter
                module = model.features[n]
                auxillery_layer = model.auxillery_layers[n]
                optimizer  = optimizers[n]
                if isinstance(module, LocalLossBlockLinear) or isinstance(module, LocalLossBlockConv):
                    if counter == len(model.features) - 2 and isinstance(module, LocalLossBlockLinear):
                        h = h.view(h.size(0), -1)
                    h, h_return = module(h) 
                    loss = loss_calc(h, h_return, y, y_onehot, module, auxillery_layer) 
                    h = h_return
                elif isinstance(module, nn.MaxPool2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Identity):
                    h = module(h)
            output = h
            test_loss += F.cross_entropy(output, target).item() * data.size(0)
        pred = output.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target_).cpu().sum()
    
    # Format and print debug string
    loss_average = test_loss / len(test_loader.dataset)
    if args.loss_sup == 'predsim' and not args.backprop:
        loss_average *= (1 - args.beta)
    error_percent = 100 - 100.0 * float(correct) / len(test_loader.dataset)

    string_print = 'Test loss_global={:.4f}, error={:.3f}%\n'.format(loss_average, error_percent)
    wandb.log({"Test Loss Global": loss_average, "Error": error_percent})
    if not args.no_print_stats:
        for m in model.modules():
            if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                string_print += m.print_stats()                
    print(string_print)
    
    return loss_average, error_percent, string_print
    
   
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
    model = VGGn(args.model, input_dim, input_ch, num_classes, args.feat_mult)
else:
    print('No valid model defined')

# Check if to load model
if checkpoint is not None:
    model.load_state_dict(checkpoint['state_dict'])
    args = args_backup
    
if args.cuda:
    model.cuda()
wandb.watch(model)
if args.progress_bar:
    from tqdm import tqdm
if not args.backprop:
    optimizers = len(model.features)*[None]
    for counter in range(len(model.features)):
        module = model.features[counter]
        auxillery_layer = model.auxillery_layers[counter]
        # temp, make sure to do it for any part that has parameters
        if isinstance(module, LocalLossBlockLinear) or isinstance(module, LocalLossBlockConv) or isinstance(module, nn.Linear): 
            to_train = itertools.chain(module.parameters(), auxillery_layer.parameters())
            if args.optim == 'sgd':
                optimizers[counter] = optim.SGD(to_train, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
            elif args.optim == 'adam' or args.optim == 'amsgrad':
                optimizers[counter] = optim.Adam(to_train, lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.optim == 'amsgrad')
            else:
                print('Unknown optimizer')
else:
    optimizers = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.optim == 'amsgrad')
    
print(model)
print('Model {} has {} parameters influenced by global loss'.format(args.model, count_parameters(model)))

''' The main training and testing loop '''
start_epoch = 1 if checkpoint is None else 1 + checkpoint['epoch']
#start_epoch = 3
print(args.epochs, start_epoch)
#args.epochs = 1
for epoch in range(start_epoch, args.epochs + 1):#(0, 2):##(start_epoch, args.epochs + 1):#range(0, 1):#
    # Decide learning rate
    lr = args.lr * args.lr_decay_fact ** bisect_right(args.lr_decay_milestones, (epoch-1))
    save_state_dict = False
    for ms in args.lr_decay_milestones:
        if (epoch-1) == ms:
            print('Decaying learning rate to {}'.format(lr))
            decay = True
        elif epoch == ms:
            save_state_dict = True

    # Set learning rate
    if not args.backprop:
        for counter, module in enumerate(model.features):
            if optimizers[counter] is not None:
                for param_group in optimizers[counter].param_groups:
                    param_group['lr'] = lr
    else:
        for param_group in optimizers.param_groups:
            param_group['lr'] = lr
    
    # Train and test    
    print(epoch, lr)
#    train(epoch, lr)
    train_loss,train_error,train_print = train(epoch, lr)

    #return
    test_loss,test_error,test_print = test(epoch)

    # Check if to save checkpoint
    if args.save_dir is not '':
        # Resolve log folder and checkpoint file name
        filename = 'chkp_ep{}_lr{:.2e}_trainloss{:.2f}_testloss{:.2f}_trainerr{:.2f}_testerr{:.2f}.tar'.format(
                epoch, lr, train_loss, test_loss, train_error, test_error)
        dirname = os.path.join(args.save_dir, args.dataset)
        dirname = os.path.join(dirname, '{}_mult{:.1f}'.format(args.model, args.feat_mult))
        dirname = os.path.join(dirname, '{}_{}x{}_{}_dimdec{}_beta{}_bs{}_drop{}_{}_wd{}_bp{}_lr{:.2e}'.format(
                args.nonlin, args.num_layers, args.num_hidden, args.loss_sup + args.loss_sup,  args.dim_in_decoder, args.beta, args.batch_size, args.dropout, args.optim, args.weight_decay, int(args.backprop), args.lr))
        
        # Create log directory
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        elif epoch==1 and os.path.exists(dirname):
            # Delete old files
            for f in os.listdir(dirname):
                os.remove(os.path.join(dirname, f))
        
        # Add log entry to log file
        with open(os.path.join(dirname, 'log.txt'), 'a') as f:
            if epoch == 1:
                f.write('{}\n\n'.format(args))
                f.write('{}\n\n'.format(model))
                f.write('{}\n\n'.format(optimizers[-1]))
                f.write('Model {} has {} parameters influenced by global loss\n\n'.format(args.model, count_parameters(model)))
            f.write(train_print)
            f.write(test_print)
            f.write('\n')
            f.close()
        
        # Save checkpoint for every epoch
        torch.save({
            'epoch': epoch,
            'args': args,
            'state_dict': model.state_dict() if (save_state_dict or epoch==args.epochs) else None,
            'train_loss': train_error,
            'train_error': train_error,
            'test_loss': test_loss,
            'test_error': test_error,
        }, os.path.join(dirname, filename))  
    
        # Save checkpoint for last epoch with state_dict (for resuming)
        torch.save({
            'epoch': epoch,
            'args': args,
            'state_dict': model.state_dict(),
            'train_loss': train_error,
            'train_error': train_error,
            'test_loss': test_loss,
            'test_error': test_error,
        }, os.path.join(dirname, 'chkp_last_epoch.tar')) 
   
