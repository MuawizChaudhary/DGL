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
from bisect import bisect_right
from nokland_utils import to_one_hot, similarity_matrix
import wandb

#### Some helper functions
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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


#####
def lr_scheduler(lr_0, epoch):
    lr = lr_0 * .25  ** bisect_right([200, 300, 350,375], epoch)
    #lr = lr_0*0.2**(epoch // 15)
    return lr

def loss_calc(outputs, y, y_onehot, module, auxillery):
    # Calculate supervised loss
    if args.loss_sup == 'sim':
        Ry = similarity_matrix(y_onehot).detach()
        loss_sup = F.mse_loss(outputs, Ry)
        if not args.no_print_stats:
            module.loss_sim += loss_sup.item() * y.size(0)
            module.examples += y.size(0)
    elif args.loss_sup == 'pred':
        loss_sup = F.cross_entropy(outputs,  y.detach())
        if not args.no_print_stats and not isinstance(module, nn.Linear):
            module.loss_pred += loss_sup.item() * y.size(0)
            module.correct += outputs.max(1)[1].eq(y).cpu().sum()
            module.examples += y.size(0)
    elif args.loss_sup == 'predsim':                    
        Rh, y_hat_local = outputs
        Ry = similarity_matrix(y_onehot).detach()
        if args.cuda:
           Ry = Ry.cuda()
        if not isinstance(module, nn.Linear):
           loss_pred = (1-args.beta) * F.cross_entropy(y_hat_local,  y.detach())
           loss_sim = args.beta * F.mse_loss(Rh, Ry)
           loss_sup = loss_pred + loss_sim
        else:
           loss_sup = F.cross_entropy(y_hat_local,  y.detach())

        if not args.no_print_stats and not isinstance(module, nn.Linear):
            module.loss_pred += loss_pred.item() * y.size(0)
            module.loss_sim += loss_sim.item() * y.size(0)
            module.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
            module.examples += y.size(0)
    return loss_sup


##################### Logs
def main():
    global args, best_prec1
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    wandb.init(config=args, project="dgl-refactored")
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if args.cuda:
        cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    time_stamp = str(datetime.datetime.now().isoformat())
    name_log_txt = time_stamp + str(randint(0, 1000)) + args.name
    name_log_txt=name_log_txt +'.log'
    
    with open(name_log_txt, "a") as text_file:
        print(args, file=text_file)
    

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284)),
    ])
    
    trainset_class = CIFAR10(root='.', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset_class, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testset = CIFAR10(root='.', train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)



    model = VGGn('vgg8b', 32, 3, 10, 1)
    #DGL_Net(aux_type=args.type_aux, block_size=args.block_size)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        model = model.cuda()
    wandb.watch(model)
    print(model)
    
    ncnn = len(model.main_cnn.blocks)
    n_cnn = len(model.main_cnn.blocks)
    with open(name_log_txt, "a") as text_file:
        print(model, file=text_file)
    ############### Initialize all
    layer_optim = [None] * ncnn
    layer_lr = [args.lr] * ncnn
    #print(len(list(model.main_cnn.blocks[0].parameters())), len(list(model.auxillary_nets[0].parameters())))
    for n in range(ncnn):
        to_train = itertools.chain(model.main_cnn.blocks[n].parameters(), model.auxillary_nets[n].parameters())
        if len(list(to_train)) != 0:
            to_train = itertools.chain(model.main_cnn.blocks[n].parameters(), model.auxillary_nets[n].parameters())
            layer_optim[n] = optim.Adam(to_train, lr=layer_lr[n])#, weight_decay=5e-4)
        else:
            layer_optim[n] = None

######################### Lets do the training
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        criterion.cuda()
    for epoch in range(1, args.epochs+1):
        # Make sure we set the bn right
        model.train()

        #For each epoch let's store each layer individually
        batch_time = [AverageMeter() for _ in range(n_cnn)]
        batch_time_total = AverageMeter()
        data_time = AverageMeter()
        losses = [AverageMeter() for _ in range(n_cnn)]
        top1 = [AverageMeter() for _ in range(n_cnn)]


        for n in range(ncnn):
            layer_lr[n] = lr_scheduler(args.lr, epoch-1)
            optimizer = layer_optim[n]
            if optimizer is not None: 
                for param_group in optimizer.param_groups:
                    param_group['lr'] = layer_lr[n]
        end = time.time()

        for i, (inputs, targets) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if args.cuda:
                targets = targets.cuda(non_blocking = True)
                inputs = inputs.cuda(non_blocking = True)
            inputs = torch.autograd.Variable(inputs)
            targets = torch.autograd.Variable(targets)


            #Main loop
            representation = inputs
            end_all = time.time()
            for n in range(ncnn):
                end = time.time()
                # Forward
                if layer_optim[n] is not None:
                    layer_optim[n].zero_grad()

                outputs, representation = model(representation, n=n)

                if layer_optim[n] is not None and not isinstance(model.main_cnn.blocks[n], nn.Linear):
                    loss = loss_calc(outputs, targets, to_one_hot(targets),
                            model.main_cnn.blocks[n], model.auxillary_nets[n])
                    #loss = criterion(outputs, targets)
                    loss.backward()
                    layer_optim[n].step()  
                if isinstance(model.main_cnn.blocks[n], nn.Linear):
                   loss = criterion(representation, targets)
                   loss.backward()
                   layer_optim[n].step()  
                representation = representation.detach()
                # measure accuracy and record loss
                # measure elapsed time
                batch_time[n].update(time.time() - end)
                if layer_optim[n] is not None:
                    if isinstance(model.main_cnn.blocks[n], nn.Linear):
                        outputs = representation
                    else:
                        outputs = outputs[1]
                    prec1 = accuracy(outputs.data, targets)
                    losses[n].update(float(loss.item()), float(inputs.size(0)))
                    top1[n].update(float(prec1[0]), float(inputs.size(0)))
        for n in range(ncnn):
            ##### evaluate on validation set
            if layer_optim[n] is not None:
                top1test = validate(val_loader, model, criterion, epoch, n)
                with open(name_log_txt, "a") as text_file:
                    print("n: {}, epoch {}, loss: {:.5f}, train top1:{} test top1:{} "
                          .format(n+1, epoch, losses[n].avg, top1[n].avg,top1test), file=text_file)

def validate(val_loader, model, criterion, epoch, n):
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

            loss = criterion(output, target)
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
