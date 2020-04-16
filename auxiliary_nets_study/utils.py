import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import datasets, transforms
from settings import parse_args
from bisect import bisect_right
import itertools
args = parse_args()

def count_parameters(model):
    ''' Count number of parameters in model influenced by global loss. '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_one_hot(y, n_dims=None):
    ''' Take integer tensor y with n dims and convert it to 1-hot representation with n+1 dims. '''
    y_tensor = y.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot

def similarity_matrix(x):
    ''' Calculate adjusted cosine similarity matrix of size x.size(0) x x.size(0). '''
    if x.dim() == 4:
        if not args.no_similarity_std and x.size(1) > 3 and x.size(2) > 1:
            z = x.view(x.size(0), x.size(1), -1)
            x = z.std(dim=2)
        else:
            x = x.view(x.size(0),-1)
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc**2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1,0)).clamp(-1,1)
    return R


def dataset_load(kwargs):
    if args.dataset == 'CIFAR10':
        input_dim = 32
        input_ch = 3
        num_classes = 10
        train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
            ])
        dataset_train = datasets.CIFAR10('../data/CIFAR10', train=True, download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            sampler = None,
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data/CIFAR10', train=False, 
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
                ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)
        return input_dim, input_ch, num_classes, train_transform, dataset_train, train_loader, test_loader

    else:
        print('No valid dataset is specified')
    
def allclose_test(output, epoch, index):
    path = "torch_tensor_" + str(epoch) + "_" + str(index) + ".pt"
    if os.path.exists(path):
        if torch.allclose(torch.load(path), output):
            print("We are allclose")
        else:
            print("YOU MADE A MISTAKE")
    else:
        torch.save(output, path)

def outputs_test(output, path):
    if os.path.exists(path):
        if torch.allclose(torch.load(path), output):
            print("We are allclose")
        else:
            print("YOU MADE A MISTAKE")
    else:
        torch.save(output, path)


def loss_calc(outputs, y, y_onehot, module):
    # Calculate supervised loss
    if args.loss_sup == 'sim':
        Ry = similarity_matrix(y_onehot).detach()
        loss_sup = F.mse_loss(outputs, Ry)
    elif args.loss_sup == 'pred':
        loss_sup = F.cross_entropy(outputs,  y.detach())
    elif args.loss_sup == 'predsim':                    
        if not isinstance(module, nn.Linear):
           Rh, y_hat_local = outputs
           Ry = similarity_matrix(y_onehot).detach()
           loss_pred = (1-args.beta) * F.cross_entropy(y_hat_local,  y.detach())
           loss_sim = args.beta * F.mse_loss(Rh, Ry)
           loss_sup = loss_pred + loss_sim
        else:
           y_hat_local = outputs
           if type(y_hat_local) == tuple:
              y_hat_local=y_hat_local[1]
           loss_sup =  F.cross_entropy(y_hat_local,  y.detach())
    return loss_sup

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

def lr_scheduler(epoch):
    lr = args.lr * args.lr_decay_fact ** bisect_right(args.lr_decay_milestones, (epoch))
    return lr

def optim_init(ncnn, model, lr):
    layer_optim = [None] * ncnn
    layer_lr = [args.lr] * ncnn
    #print(len(list(model.main_cnn.blocks[0].parameters())), len(list(model.auxillary_nets[0].parameters())))
    for n in range(ncnn):
        to_train = itertools.chain(model.main_cnn.blocks[n].parameters(), model.auxillary_nets[n].parameters())
        if len(list(to_train)) != 0:
            to_train = itertools.chain(model.main_cnn.blocks[n].parameters(), model.auxillary_nets[n].parameters())
            layer_optim[n] = optim.Adam(to_train, lr=layer_lr[n], weight_decay=args.weight_decay, amsgrad=args.optim == 'amsgrad')#, weight_decay=5e-4)
        else:
            layer_optim[n] = None
    return layer_optim, layer_lr


