import torch
import os
from torchvision import datasets, transforms
from settings import parse_args

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
    
