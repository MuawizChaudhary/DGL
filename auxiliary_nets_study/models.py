import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math


class rep(nn.Module):
    def __init__(self, blocks):
        super(rep, self).__init__()
        self.blocks = blocks
    def forward(self, x, n, upto=False):
        # if upto = True we forward from the input to output of layer n
        # if upto = False we forward just through layer n
        if upto:
            for i in range(n+1):
                x = self.forward(x,i,upto=False)
            return x
        out = self.blocks[n](x)
        return out
      
class Net(nn.Module):
    def __init__(self, depth=6, num_classes=10, aux_type='mlp', block_size=1,
                 feature_size=128, downsample=None, biological=False):
        super(Net, self).__init__()

        if aux_type == 'mlp':
            nlin=0; mlp_layers=3; cnn=False
        elif aux_type == 'mlp-sr':
            nlin=3; mlp_layers=3; cnn=False
        elif aux_type == 'cnn':
            nlin=2; mlp_layers=0; cnn = True

        if downsample is None:
            downsample = [1,3]

        self.blocks = nn.ModuleList([])
        self.auxillary_nets = nn.ModuleList([])
        self.in_size = 32

        self.blocks.append(nn.Sequential(
            nn.Conv2d(3, feature_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_size), nn.ReLU()
        ))

        self.auxillary_nets.append(
            auxillary_classifier3(input_features=feature_size, cnn = cnn,
                                  in_size=self.in_size, num_classes=num_classes,
                                  n_lin=nlin, mlp_layers=mlp_layers, biological=biological))
        
        for i in range(depth-1):
            if i+1 in downsample:
                self.blocks.append(nn.Sequential(
                    nn.MaxPool2d((2,2)),
                    nn.Conv2d(feature_size, feature_size*2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature_size*2), nn.ReLU()
                ))
                self.in_size/=2
                feature_size*=2
            else:
                self.blocks.append(nn.Sequential(
                    nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature_size), nn.ReLU()
                ))


            if i < depth-2:
                self.auxillary_nets.append(
                    auxillary_classifier3(input_features=feature_size, cnn = cnn,
                                          in_size=self.in_size, num_classes=num_classes,
                                          n_lin=nlin,mlp_layers=mlp_layers, biological=biological))

        self.auxillary_nets.append(auxillary_classifier3(input_features=feature_size,
                                          in_size=self.in_size, num_classes=num_classes,
                                          n_lin=0,mlp_layers=2, biological=biological))

        self.main_cnn = rep(self.blocks)
        
    def forward(self, representation, n, upto=False):
        representation = self.main_cnn.forward(representation, n, upto=upto)
        outputs = self.auxillary_nets[n](representation)
        return outputs, representation



class auxillary_classifier2(nn.Module):
    def __init__(self, input_features=256, in_size=32, cnn = False,
                 num_classes=10, n_lin=0, mlp_layers=0):
        super(auxillary_classifier2, self).__init__()
        self.n_lin=n_lin
        self.in_size=in_size
        self.cnn = cnn
        feature_size = input_features
        self.blocks = []
        for n in range(self.n_lin):
            if n==0:
                input_features = input_features
            else:
                input_features = feature_size

            bn_temp = nn.BatchNorm2d(feature_size)

            if cnn:
                conv = nn.Conv2d(input_features, feature_size,
                                 kernel_size=3, stride=1, padding=1, bias=false)
            else:
                conv = nn.Conv2d(input_features, feature_size,
                                 kernel_size=1, stride=1, padding=0, bias=false)
            self.blocks.append(nn.sequential(conv,bn_temp))

        self.blocks = nn.ModuleList(self.blocks)
        self.bn = nn.BatchNorm2d(feature_size)

        if mlp_layers > 0:

            mlp_feat = feature_size * (2) * (2)
            layers = []

            for l in range(mlp_layers):
                if l==0:
                    in_feat = feature_size*4
                    mlp_feat = mlp_feat
                else:
                    in_feat = mlp_feat

                bn_temp = nn.batchnorm1d(mlp_feat)


                layers +=[nn.linear(in_feat,mlp_feat),
                              bn_temp,nn.relu(true)]
            layers += [nn.linear(mlp_feat,num_classes)]
            self.classifier = nn.sequential(*layers)
            self.mlp = true

        else:
            self.mlp = false
            self.classifier = nn.linear(feature_size*4, num_classes)


    def forward(self, x):
        out = x
        if not self.cnn:
            #first reduce the size by 16
            out = f.adaptive_avg_pool2d(out,(math.ceil(self.in_size/4),math.ceil(self.in_size/4)))

        for n in range(self.n_lin):
            out = self.blocks[n](out)
            out = f.relu(out)

        out = f.adaptive_avg_pool2d(out, (2,2))
        if not self.mlp:
            out = self.bn(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class auxillary_classifier3(nn.Module):
    def __init__(self, input_features=256, in_size=32, cnn=False,
                 num_classes=10, n_lin=0, mlp_layers=0, biological=True):
        super(auxillary_classifier3, self).__init__()
        self.n_lin=n_lin
        self.in_size=in_size
        self.cnn = cnn
        feature_size = input_features
        self.blocks = []
        for n in range(self.n_lin):
            if n==0:
                input_features = input_features
            else:
                input_features = feature_size

            bn_temp = nn.BatchNorm2d(feature_size)

            if cnn:
                conv = nn.Conv2d(input_features, feature_size,
                                 kernel_size=3, stride=1, padding=1, bias=False)
            else:
                conv = nn.Conv2d(input_features, feature_size,
                                 kernel_size=1, stride=1, padding=0, bias=False)
            self.blocks.append(nn.Sequential(conv,bn_temp))

        self.blocks = nn.ModuleList(self.blocks)
        self.bn = nn.BatchNorm2d(feature_size)

        if mlp_layers > 0:

            mlp_feat = feature_size * (2) * (2)
            layers = []

            for l in range(mlp_layers):
                if l==0:
                    in_feat = feature_size*4
                    mlp_feat = mlp_feat
                else:
                    in_feat = mlp_feat

                bn_temp = nn.BatchNorm1d(mlp_feat)


                layers +=[nn.Linear(in_feat,mlp_feat),
                              bn_temp,nn.ReLU(True)]
            layers += [nn.Linear(mlp_feat,num_classes)]
            self.classifier = nn.Sequential(*layers)
            self.mlp = True
        else:
            self.mlp = False
            self.classifier = nn.Linear(int(input_features*in_size*in_size), num_classes)
        print(input_features, in_size)
        in_size=int(in_size)

        if biological:#featuresize(5
            self.decoder_y = LinearFA(int(input_features*in_size*in_size), num_classes)
            self.proj_y = nn.Linear(num_classes, num_classes, bias=False)
        else:
            print(input_features*in_size*in_size,int(num_classes))
            self.decoder_y = nn.Linear(int(input_features*in_size*in_size), int(num_classes))
            self.linear_loss = nn.Linear(in_size, in_size, bias=False)
        self.decoder_y.weight.data.zero_()
    
    def forward(self, x):
        out = x
        if not self.cnn:
            #first reduce the size by 16
            out = F.adaptive_avg_pool2d(out,(math.ceil(self.in_size/4),math.ceil(self.in_size/4)))

        for n in range(self.n_lin):
            out = self.blocks[n](out)
            out = F.relu(out)

        out = F.adaptive_avg_pool2d(out, (2,2))
        if not self.mlp:
            out = self.bn(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


#self.conv_loss = nn.Conv2d(ch_out, ch_out, 3, stride=1, padding=1, bias=False)




class LinearFAFunction(torch.autograd.Function):
    '''Autograd function for linear feedback alignment module.
    '''
    @staticmethod
    def forward(context, input, weight, weight_fa, bias=None):
        context.save_for_backward(input, weight, weight_fa, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_fa, bias = context.saved_variables
        grad_input = grad_weight = grad_weight_fa = grad_bias = None

        if context.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight_fa)
        if context.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
        if bias is not None and context.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_weight_fa, grad_bias

class LinearFA(nn.Module):
    '''Linear feedback alignment module.

    Args:
        input_features (int): Number of input features to linear layer.
        output_features (int): Number of output features from linear layer.
        bias (bool): True if to use trainable bias.
    '''
    def __init__(self, input_features, output_features, bias=True):
        super(LinearFA, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.weight_fa = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
        if torch.cuda.is_available:
            self.weight.data = self.weight.data.cuda()
            self.weight_fa.data = self.weight_fa.data.cuda()
            if bias:
                self.bias.data = self.bias.data.cuda()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_fa.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()
            
    def forward(self, input):
        return LinearFAFunction.apply(input, self.weight, self.weight_fa, self.bias)
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.input_features) \
            + ', out_features=' + str(self.output_features) \
            + ', bias=' + str(self.bias is not None) + ')'
 


 


#if not args.backprop and (args.loss_sup == 'pred' or args.loss_sup == 'predsim'):
#            if args.bio:
#                self.decoder_y = LinearFA(num_out, num_classes)
#            else:
#                self.decoder_y = nn.Linear(num_out, num_classes)
#            self.decoder_y.weight.data.zero_()
#        if not args.backprop and args.bio:
#            self.proj_y = nn.Linear(num_classes, num_classes, bias=False)
#        if not args.backprop and not args.bio and (args.loss_unsup == 'sim' or args.loss_sup == 'sim' or args.loss_sup == 'predsim'):
#            self.linear_loss = nn.Linear(num_out, num_out, bias=False)
#        if self.batchnorm:
#            self.bn = torch.nn.BatchNorm1d(num_out)
#            nn.init.constant_(self.bn.weight, 1)
#            nn.init.constant_(self.bn.bias, 0)
#
