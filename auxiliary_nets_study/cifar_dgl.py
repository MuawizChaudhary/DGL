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
from models import auxillary_classifier2, Net
from utils import to_one_hot, similarity_matrix

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
def lr_scheduler(lr_0,epoch):
    lr = lr_0*0.2**(epoch // 15)
    return lr

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--type_aux', type=str, default='mlp-sr',metavar='N')
parser.add_argument('--block_size', type=int, default=1, help='block size')
parser.add_argument('--name', default='',type=str,help='name')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-similarity-std', action='store_true', default=False,
                    help='disable use of standard deviation in similarity matrix for feature maps')
parser.add_argument('--bio', action='store_true', default=True,
                    help='use more biologically plausible versions of pred and sim loss (default: False)')
parser.add_argument('--loss-sup', default='predsim',
                    help='supervised local loss, sim or pred (default: predsim)')
parser.add_argument('--loss-unsup', default='none',
                    help='unsupervised local loss, none, sim or recon (default: none)')
parser.add_argument('--beta', type=float, default=0.99,
                    help='fraction of similarity matching loss in predsim loss (default: 0.99)')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


##################### Logs
time_stamp = str(datetime.datetime.now().isoformat())
name_log_txt = time_stamp + str(randint(0, 1000)) + args.name
name_log_txt=name_log_txt +'.log'

with open(name_log_txt, "a") as text_file:
    print(args, file=text_file)

def main():
    global args, best_prec1
    args = parser.parse_args()


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset_class = CIFAR10(root='.', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset_class, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testset = CIFAR10(root='.', train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)



    model = Net(aux_type=args.type_aux, block_size=args.block_size,
            biological=args.bio)
    print(args.block_size)
    model = model.cuda()
    
    ncnn = len(model.main_cnn.blocks)
    n_cnn = len(model.main_cnn.blocks)
    with open(name_log_txt, "a") as text_file:
        print(model, file=text_file)
    ############### Initialize all
    layer_optim = [None] * ncnn
    layer_lr = [0.1] * ncnn
    for n in range(ncnn):
        to_train = itertools.chain(model.main_cnn.blocks[n].parameters(),
                                           model.auxillary_nets[n].parameters())
        layer_optim[n] = optim.SGD(to_train, lr=layer_lr[n], momentum=0.9,  
                                  weight_decay=5e-4)

######################### Lets do the training
    criterion = nn.CrossEntropyLoss().cuda()
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
            layer_lr[n] = lr_scheduler(0.1, epoch-1)
            for param_group in layer_optim[n].param_groups:
                param_group['lr'] = layer_lr[n]
        end = time.time()

        for i, (inputs, targets) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            targets = targets.cuda(non_blocking = True)
            inputs = inputs.cuda(non_blocking = True)
            inputs = torch.autograd.Variable(inputs)
            targets = torch.autograd.Variable(targets)
            target_one_hot = to_one_hot(targets)


            #Main loop
            representation = inputs
            end_all = time.time()
            for n in range(ncnn):
                end = time.time()
                # Forward
                layer_optim[n].zero_grad()
                outputs, representation = model(representation, n=n)

                # loss calculation 
                if args.loss_sup == 'sim' or args.loss_sup == 'predsim':
                    if args.bio:
                        h_loss = representation.cuda()
                    else:
                        h_loss = model.auxillary_nets[n].linear_loss(representation).cuda()
                    Rh = similarity_matrix(h_loss)
             
                if args.loss_sup == 'sim':
                    if args.bio:
                        Ry = similarity_matrix(model.auxillary_nets[n].proj_y(target_one_hot.cuda())).cuda().detach()
                    else:
                        Ry = similarity_matrix(target_one_hot.cuda()).detach()
                    loss_sup = F.mse_loss(Rh, Ry)
                elif args.loss_sup == 'pred':
                    y_hat_local = model.auxillary_nets[n].decoder_y(representation.view(representation.size(0), -1).cuda())
                    if args.bio:
                        float_type =  torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
                        y_onehot_pred = model.auxillary_nets[n].proj_y(target_one_hot).gt(0).type(float_type).detach()
                        loss_sup = F.binary_cross_entropy_with_logits(y_hat_local, y_onehot_pred)
                    else:
                        loss_sup = F.cross_entropy(y_hat_local,  y.detach())
                elif args.loss_sup == 'predsim':                    
                    y_hat_local = model.auxillary_nets[n].decoder_y(representation.view(representation.size(0), -1))
                    if args.bio:
                        Ry = similarity_matrix(model.auxillary_nets[n].proj_y(target_one_hot.cuda()).cuda()).detach()
                        float_type =  torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
                        y_onehot_pred = self.proj_y(target_one_hot).gt(0).type(float_type).detach()
                        loss_pred = (1-args.beta) * F.binary_cross_entropy_with_logits(y_hat_local, y_onehot_pred)
                    else:
                        Ry = similarity_matrix(target_one_hot).detach().cuda()
                        loss_pred = (1-args.beta) * F.cross_entropy(y_hat_local,  targets.detach())
                    loss_sim = args.beta * F.mse_loss(Rh, Ry)
                    loss_sup = loss_pred + loss_sim
           
                #TODO Combine unsupervised and supervised loss
                loss = loss_sup

                #loss = criterion(outputs, targets)
                loss.backward()
                layer_optim[n].step()

                representation = representation.detach()
                # measure accuracy and record loss
                # measure elapsed time
                batch_time[n].update(time.time() - end)

                prec1 = accuracy(outputs.data, targets)
                losses[n].update(float(loss.data.item()), float(inputs.size(0)))
                top1[n].update(float(prec1[0]), float(inputs.size(0)))
        for n in range(ncnn):
            ##### evaluate on validation set
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
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            input = torch.autograd.Variable(input)
            target = torch.autograd.Variable(target)
            target_one_hot = to_one_hot(target)

            representation = input
            output, _ = model(representation, n=n, upto=True)


            loss = criterion(output, target)
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)
            losses.update(float(loss.data.item()), float(input.size(0)))
            top1.update(float(prec1[0]), float(input.size(0)))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            total += input.size(0)
        print(' * Prec@1 {top1.avg:.3f}'
              .format(top1=top1))


    return top1.avg


if __name__ == '__main__':
    main()

# Calculate supervised loss
            
