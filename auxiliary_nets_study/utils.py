import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import wandb

def to_one_hot(y, n_dims=None):
    ''' Take integer tensor y with n dims and convert it to 1-hot representation with n+1 dims. '''
    y_tensor = y.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot

def similarity_matrix(x, no_similarity_std):
    ''' Calculate adjusted cosine similarity matrix of size x.size(0) x x.size(0). '''
    if x.dim() == 4:
        if not no_similarity_std and x.size(1) > 3 and x.size(2) > 1:
            z = x.view(x.size(0), x.size(1), -1)
            x = z.std(dim=2)
        else:
            x = x.view(x.size(0),-1)
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc**2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1,0)).clamp(-1,1)
    return R

def loss_calc(outputs, y, y_onehot, module, loss_sup, beta, no_similarity_std):
    if not isinstance(module, nn.Linear) and type(outputs) == tuple:
        Rh, y_hat_local = outputs
    else:
        Rh = outputs
        y_hat_local = outputs
    # Calculate supervised loss
    if loss_sup == 'pred':
        loss_sup = F.cross_entropy(y_hat_local,  y)
    elif loss_sup == 'predsim':
        if not isinstance(module, nn.Linear) and not torch.equal(Rh,y_hat_local):
            #Rh, y_hat_local = outputs 
            Rh = similarity_matrix(Rh, no_similarity_std)
            Ry = similarity_matrix(y_onehot, no_similarity_std).detach()
            loss_pred = (1-beta) * F.cross_entropy(y_hat_local,  y)
            loss_sim = beta * F.mse_loss(Rh, Ry)
            loss_sup = loss_pred + loss_sim
        else:
            y_hat_local = outputs
            if type(y_hat_local) == tuple:
                y_hat_local=y_hat_local[1]
            loss_sup = (1-beta) * F.cross_entropy(y_hat_local,  y)
    return loss_sup


# Test routines

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




def test(epoch, model, test_loader, cuda=True,num_classes=10):
    ''' Run model on test set '''
    model.eval()
    correct = 0

    # Loop test set
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        target_ = target
        h, y,  = data, target

        for n in range(len(model.main_cnn.blocks)):
            output, h = model(h, n=n)

        output = h
        pred = output.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target_).cpu().sum()

    error_percent = 100.0 * float(correct) / len(test_loader.dataset)
    print('acc: ' + str(error_percent))

    return error_percent


def validate(val_loader, model, epoch, n, loss_sup, iscuda):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    with torch.no_grad():
        total = 0
        for i, (input, target) in enumerate(val_loader):
            if iscuda:
                target = target.cuda(non_blocking=True)
                input = input.cuda(non_blocking=True)

            representation = input
            for i in range(n):
                output, representation = model(representation, n=i)
                representation = representation.detach()
                # measure accuracy and record loss
                # measure elapsed time
            output, representation = model(representation, n=n)
            #if loss_sup == "predsim":
            #output = output[1]
            if isinstance(output, tuple):
                output = output[1]
            if isinstance(model.module.main_cnn.blocks[n], nn.Linear):
               output = representation

            # measure accuracy and record loss
            loss = F.cross_entropy(output, target)

            prec1 = accuracy(output.data, target)
            losses.update(float(loss.item()), float(input.size(0)))
            top1.update(float(prec1[0]), float(input.size(0)))

            total += input.size(0)
        #wandb.log({"Layer " + str(n) + " test loss": losses.avg}, step=epoch)
        #wandb.log({"Layer " + str(n) + " top1": top1.avg}, step=epoch)
    return top1.avg

from torch.nn.parallel import DataParallel
from torch.nn.parallel.scatter_gather import scatter_kwargs,scatter
class DataParallelSpecial(DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallelSpecial,self).__init__(module, device_ids=None, output_device=None, dim=0)
        print('Initialized with GPUs:')
        print(self.device_ids)

    def forward(self, *inputs, init=False, **kwargs):
        if init:
            if self.device_ids:
                # -------- Here, we split the input tensor across GPUs
                inputs_ = inputs
                if not isinstance(inputs_, tuple):
                    inputs_ = (inputs_,)

                representation, _ = scatter_kwargs(inputs_, None, self.device_ids, 0)
                self.replicas = self.replicate(self.module, self.device_ids[:len(representation)])
                # ----
            else:
                representation = inputs
            return None , representation

        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        # inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
        if len(self.device_ids) == 1:
            import ipdb; ipdb.set_trace()
            return self.module(*inputs[0][0], **kwargs)

        kwargs = scatter(kwargs, self.device_ids) if kwargs else []
         #   if len(inputs) < len(kwargs):
          #      inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
           # elif len(kwargs) < len(inputs):
            #    kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        kwargs = tuple(kwargs)
        outputs = self.parallel_apply(self.replicas, *inputs, kwargs)

        out1 = []
        out2 = []
        for i, tensor in enumerate(outputs):
            with torch.cuda.device(tensor[0].get_device()):
                # out_1[i] = torch.autograd.Variable(tensors[i])
                out1.append(outputs[i][0])
                out2.append(outputs[i][1])
        outputs = self.gather(out1, self.output_device)
        representation = out2
        return outputs, representation


