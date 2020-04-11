import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from settings import parse_args

args = parse_args()

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
                if type(x) == tuple:
                   x = x[1]
            return x
        out = self.blocks[n](x)
        return out
      
class DGL_Net(nn.Module):
    def __init__(self, depth=8, num_classes=10, aux_type='mlp', block_size=1,
                 feature_size=128, downsample=None, dropout_p=0.2):#put it back to .2
        super(Net, self).__init__()
        print(aux_type)
        if aux_type == 'mlp':
            nlin=0; mlp_layers=0; cnn=False
        elif aux_type == 'mlp-sr':
            nlin=3; mlp_layers=3; cnn=False
        elif aux_type == 'cnn':
            nlin=2; mlp_layers=0; cnn = True

        self.dropout_p = dropout_p
        self.dropout_module = torch.nn.Dropout2d
        self.num_classes = num_classes
        relu = nn.ReLU()#nn.LeakyReLU(negative_slope=0.01)

        if downsample is None:
            downsample = [2, 4, 6]

        blocks = nn.ModuleList([])
        self.auxillary_nets = nn.ModuleList([])
        self.in_size = 32

        blocks.append(nn.Sequential(
            nn.Conv2d(3, feature_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_size), relu, self.dropout_module(p=self.dropout_p, inplace=False)
        ))

        self.auxillary_nets.append(
            auxillary_classifier2(input_features=feature_size, cnn = cnn,
                                  in_size=self.in_size, num_classes=num_classes,
                                  n_lin=nlin, mlp_layers=mlp_layers))
        
        for i in range(depth-1):
            if i+1 in downsample:
                blocks.append(nn.Sequential(
                    nn.MaxPool2d((2,2)),
                    nn.Conv2d(feature_size, feature_size*2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature_size*2), relu,
                    self.dropout_module(p=self.dropout_p, inplace=False)
                ))
                self.in_size/=2
                feature_size*=2
            else:
                blocks.append(nn.Sequential(
                    nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature_size), relu, self.dropout_module(p=self.dropout_p, inplace=False)
                ))


            if i < depth-2:
                self.auxillary_nets.append(
                    auxillary_classifier2(input_features=feature_size, cnn = cnn,
                                          in_size=self.in_size, num_classes=num_classes,
                                          n_lin=nlin,mlp_layers=mlp_layers))

        self.auxillary_nets.append(auxillary_classifier2(input_features=feature_size,
                                          in_size=self.in_size, num_classes=num_classes,
                                          n_lin=nlin,mlp_layers=mlp_layers))

        if block_size>1:
            len_layers = len(blocks)
            self.block_temp = nn.ModuleList([])
            self.aux_temp = nn.ModuleList([])
            for splits_id in range(depth//block_size):
                left_idx = splits_id * block_size
                right_idx = (splits_id + 1) * block_size
                if right_idx > len_layers:
                    right_idx = len_layers
                self.block_temp.append(nn.Sequential(*blocks[left_idx:right_idx]))
                self.aux_temp.append(self.auxillary_nets[right_idx-1])
         #   self.aux_temp[len(self.aux_temp)-1] = self.auxillary_nets[len(self.auxillary_nets)-1]
            blocks = self.block_temp
            self.auxillary_nets = self.aux_temp
        blocks, auxillary_nets = self._make_layers(cfg['vgg8b'], 3, feature_size, 1)
        print(len(blocks))
        self.auxillary_nets = auxillary_nets
        self.main_cnn = rep(blocks)
        
    def forward(self, representation, n, upto=False):
        representation = self.main_cnn.forward(representation, n, upto=upto)
        if isinstance(representation, tuple):
            representation = representation[1]
        outputs = self.auxillary_nets[n](representation)
        return outputs, representation

    def _make_layers(self, cfg, input_ch, input_dim, feat_mult):
        layers = []
        auxillery_layers = []
        first_layer = True
        scale_cum = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                auxillery_layers += [nn.Identity(1)]
                scale_cum *=2
            elif x == 'M3':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
                auxillery_layers += [nn.Identity()]
                scale_cum *=2
            elif x == 'M4':
                layers += [nn.MaxPool2d(kernel_size=4, stride=4)]
                auxillery_layers += [nn.Identity()]
                scale_cum *=4
            elif x == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                auxillery_layers += [nn.Identity()]
                scale_cum *=2
            elif x == 'A3':
                layers += [nn.AvgPool2d(kernel_size=3, stride=2, padding=1)]
                auxillery_layers += [nn.Identity()]
                scale_cum *=2
            elif x == 'A4':
                layers += [nn.AvgPool2d(kernel_size=4, stride=4)]
                auxillery_layers += [nn.Identity()]
                scale_cum *=4
            else:
                x = int(x * feat_mult)
                if first_layer and input_dim > 64:
                    scale_cum = 2
                    layers += [LocalLossBlockConv(input_ch, x, kernel_size=7, stride=2, padding=3, 
                                             num_classes=self.num_classes, 
                                             dim_out=input_dim//scale_cum, 
                                             first_layer=first_layer, dropout=self.dropout_p)]
                    auxillery_layers += [auxillary_classifier2(x)]

                else:
                    layers += [LocalLossBlockConv(input_ch, x, kernel_size=3, stride=1, padding=1, 
                                             num_classes=self.num_classes, 
                                             dim_out=input_dim//scale_cum, 
                                             first_layer=first_layer, dropout=self.dropout_p)]
                    auxillery_layers += [auxillary_classifier2(x)]

                input_ch = x
                first_layer = False
        output_dim=input_dim//scale_cum
        layers += [LocalLossBlockConv(input_ch, 1024, kernel_size=3, stride=1, padding=1, 
                                             num_classes=self.num_classes, 
                                             dim_out=input_dim//scale_cum, 
                                             first_layer=first_layer, dropout=self.dropout_p)]
        auxillery_layers += [auxillary_classifier2(1024)]
        return nn.ModuleList(layers), nn.ModuleList(auxillery_layers)


class auxillary_classifier2(nn.Module):
    def __init__(self, input_features=256, in_size=32, cnn = False,
                 num_classes=10, n_lin=0, mlp_layers=0):
        super(auxillary_classifier2, self).__init__()
        self.n_lin=n_lin
        self.in_size=in_size
        self.cnn = cnn
        feature_size = input_features
        self.blocks = []
        #TODO make argument
        #self.dropout_p = 0.2
        #self.dropout = torch.nn.Dropout2d(p=self.dropout_p, inplace=False)

        
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
        #self.relu = nn.ReLU()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((2,2))
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
            self.classifier = nn.Linear(feature_size*4, num_classes)


    def forward(self, x):
        if type(x) is tuple:
            x = x[1]
        out = x
        if not self.cnn:
            #First reduce the size by 16
            out = F.adaptive_avg_pool2d(out,(math.ceil(self.in_size/4),math.ceil(self.in_size/4)))


        for n in range(self.n_lin):
            out = self.blocks[n](out)
            #out = self.relu(out)
            #out = self.dropout(out)

        out = self.adaptive_avg_pool(out)

        if not self.mlp:
            out = self.bn(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class LocalLossBlockLinear(nn.Module):
    '''A module containing nn.Linear -> nn.BatchNorm1d -> nn.ReLU -> nn.Dropout
       The block can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
       
    Args:
        num_in (int): Number of input features to linear layer.
        num_out (int): Number of output features from linear layer.
        num_classes (int): Number of classes (used in local prediction loss).
        first_layer (bool): True if this is the first layer in the network (used in local reconstruction loss).
        dropout (float): Dropout rate, if None, read from args.dropout.
    '''
    def __init__(self, num_in, num_out, num_classes, first_layer=False, dropout=None, batchnorm=None):
        super(LocalLossBlockLinear, self).__init__()
        self.num_classes = num_classes
        self.first_layer = first_layer
        self.dropout_p = args.dropout if dropout is None else dropout
        encoder = nn.Linear(num_in, num_out, bias=True)
        
        bn = torch.nn.BatchNorm1d(num_out)
        nn.init.constant_(bn.weight, 1)
        nn.init.constant_(bn.bias, 0)          

        if args.nonlin == 'relu':
            nonlin = nn.ReLU(inplace=True)
        elif args.nonlin == 'leakyrelu':
            nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        self.MLP = nn.Sequential(encoder, bn, nonlin)
        
        if self.dropout_p > 0:
            self.dropout = torch.nn.Dropout(p=self.dropout_p, inplace=False)
           
        self.clear_stats()
    
    def clear_stats(self):
        if not args.no_print_stats:
            self.loss_sim = 0.0
            self.loss_pred = 0.0
            self.correct = 0
            self.examples = 0

    def print_stats(self):
        if not args.backprop:
            stats = '{}, loss_sim={:.4f}, loss_pred={:.4f}, error={:.3f}%, num_examples={}\n'.format(
                    self.MLP[0],
                    self.loss_sim / self.examples, 
                    self.loss_pred / self.examples,
                    100.0 * float(self.examples - self.correct) / self.examples,
                    self.examples)
            return stats
        else:
            return ''
    
    def forward(self, x):        
        h = self.MLP(x)
        h_return = h
        if self.dropout_p > 0:
            h_return = self.dropout(h_return)
        return h, h_return

    
class LocalLossBlockConv(nn.Module):
    '''
    A block containing nn.Conv2d -> nn.BatchNorm2d -> nn.ReLU -> nn.Dropou2d
    The block can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
    
    Args:
        ch_in (int): Number of input features maps.
        ch_out (int): Number of output features maps.
        kernel_size (int): Kernel size in Conv2d.
        stride (int): Stride in Conv2d.
        padding (int): Padding in Conv2d.
        num_classes (int): Number of classes (used in local prediction loss).
        dim_out (int): Feature map height/width for input (and output).
        first_layer (bool): True if this is the first layer in the network (used in local reconstruction loss).
        dropout (float): Dropout rate, if None, read from args.dropout.
        bias (bool): True if to use trainable bias.
    '''
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding, num_classes, dim_out, first_layer=False, dropout=None):
        super(LocalLossBlockConv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.num_classes = num_classes
        self.first_layer = first_layer
        self.dropout_p = args.dropout if dropout is None else dropout
        encoder = nn.Conv2d(ch_in, ch_out, kernel_size, stride=stride, padding=padding)
        
        bn = torch.nn.BatchNorm2d(ch_out)
        nn.init.constant_(bn.weight, 1)
        nn.init.constant_(bn.bias, 0)

        if args.nonlin == 'relu':
            nonlin = nn.ReLU(inplace=True)
        elif args.nonlin == 'leakyrelu':
            nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        self.MLP = nn.Sequential(encoder, bn, nonlin)

        if self.dropout_p > 0:
            self.dropout = torch.nn.Dropout2d(p=self.dropout_p, inplace=False)
        self.clear_stats()
    
    def clear_stats(self):
        if not args.no_print_stats:
            self.loss_sim = 0.0
            self.loss_pred = 0.0
            self.correct = 0
            self.examples = 0

    def print_stats(self):
        if not args.backprop:
            stats = '{}, loss_sim={:.4f}, loss_pred={:.4f}, error={:.3f}%, num_examples={}\n'.format(
                    self.MLP[0],
                    self.loss_sim / self.examples, 
                    self.loss_pred / self.examples,
                    100.0 * float(self.examples - self.correct) / self.examples,
                    self.examples)
            return stats
        else:
            return ''
    
    def forward(self, x):
        h = self.MLP(x)
        h_return = h
        if self.dropout_p > 0:
            h_return = self.dropout(h_return)
        return h, h_return

class Auxillery(nn.Module):
    def __init__(self, auxillery_linear, dim_out, num_classes, num_out):
        super(Auxillery, self).__init__()
        if auxillery_linear == "conv":
            if (not args.backprop and (args.loss_sup == 'pred' or args.loss_sup == 'predsim')):
                # Resolve average-pooling kernel size in order for flattened dim to match args.dim_in_decoder
                ks_h, ks_w = 1, 1
                dim_out_h, dim_out_w = dim_out, dim_out
                dim_in_decoder = num_out*dim_out_h*dim_out_w
                print(dim_in_decoder, num_out, dim_out, dim_out_w)
                while dim_in_decoder > args.dim_in_decoder and ks_h < dim_out:
                    ks_h*=2
                    dim_out_h = math.ceil(dim_out / ks_h)
                    dim_in_decoder = num_out*dim_out_h*dim_out_w
                    if dim_in_decoder > args.dim_in_decoder:
                       ks_w*=2
                       dim_out_w = math.ceil(dim_out / ks_w)
                       dim_in_decoder = num_out*dim_out_h*dim_out_w 
                if ks_h > 1 or ks_w > 1:
                    pad_h = (ks_h * (dim_out_h - dim_out // ks_h)) // 2
                    pad_w = (ks_w * (dim_out_w - dim_out // ks_w)) // 2
                    self.avg_pool = nn.AvgPool2d((ks_h,ks_w), padding=(pad_h, pad_w))
                else:
                    self.avg_pool = nn.Identity()
            if not args.backprop and (args.loss_sup == 'pred' or args.loss_sup == 'predsim'):
                print(dim_in_decoder, num_classes)
                self.decoder_y = nn.Linear(dim_in_decoder, num_classes)
                self.decoder_y.weight.data.zero_()
            if not args.backprop and (args.loss_sup == 'sim' or args.loss_sup == 'predsim'):
                self.sim_loss = nn.Conv2d(num_out, num_out, 3, stride=1, padding=1, bias=False) 
        else:
            self.avg_pool = nn.Identity()
            if not args.backprop and (args.loss_sup == 'pred' or args.loss_sup == 'predsim'):
                self.decoder_y = nn.Linear(num_out, num_classes)
                self.decoder_y.weight.data.zero_()
            if not args.backprop and (args.loss_sup == 'sim' or args.loss_sup == 'predsim'):
                self.sim_loss = nn.Linear(num_out, num_out, bias=False)


class Net(nn.Module):
    '''
    A fully connected network.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
    
    Args:
        num_layers (int): Number of hidden layers.
        num_hidden (int): Number of units in each hidden layer.
        input_dim (int): Feature map height/width for input.
        input_ch (int): Number of feature maps for input.
        num_classes (int): Number of classes (used in local prediction loss).
    '''
    def __init__(self, num_layers, num_hidden, input_dim, input_ch, num_classes):
        super(Net, self).__init__()
        
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        reduce_factor = 1
        self.layers = nn.ModuleList([LocalLossBlockLinear(input_dim*input_dim*input_ch, num_hidden, num_classes, first_layer=True)])
        self.auxillery_layers = nn.ModuleList([Auxillery("linear", num_hidden, num_classes, num_hidden)]) 
        self.layers.extend([LocalLossBlockLinear(int(num_hidden // (reduce_factor**(i-1))), int(num_hidden // (reduce_factor**i)), num_classes) for i in range(1, num_layers)])
        self.auxillery_layers.extend([Auxillery("linear", int(num_hidden // (reduce_factor**i)), num_classes, int(num_hidden // (reduce_factor**i))) for i in range(1, num_layers)])
        
        layer_out = nn.Linear(int(num_hidden //(reduce_factor**(num_layers-1))), num_classes)
        if not args.backprop:
            layer_out.weight.data.zero_()
        
        self.layers.extend([layer_out])
        self.auxillery_layers.extend([nn.Identity(1)])
            
    def parameters(self):
        if not args.backprop:
            return self.layer_out.parameters()
        else:
            return super(Net, self).parameters()
#    
           
cfg = {
    'vgg6a':  [128, 'M', 256, 'M', 512, 'M', 512],
    'vgg6b':  [128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'vgg8':   [ 64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'vgg8a':  [128, 256, 'M', 256, 512, 'M', 512, 'M', 512],
    'vgg8b':  [128, 256, 'M', 256, 512, 'M', 512, 'M', 512, 'M'],
    'vgg11':  [ 64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg11b': [128, 128, 128, 256, 'M', 256, 512, 'M', 512, 512, 'M', 512, 'M'],
    'vgg13':  [ 64,  64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16':  [ 64,  64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19':  [ 64,  64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGn(nn.Module):
    '''
    VGG and VGG-like networks.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
    
    Args:
        vgg_name (str): The name of the network.
        input_dim (int): Feature map height/width for input.
        input_ch (int): Number of feature maps for input.
        num_classes (int): Number of classes (used in local prediction loss).
        feat_mult (float): Multiply number of feature maps with this number.
    '''
    def __init__(self, vgg_name, input_dim, input_ch, num_classes, feat_mult=1):
        super(VGGn, self).__init__()
        self.cfg = cfg[vgg_name]
        self.input_dim = input_dim
        self.input_ch = input_ch
        self.num_classes = num_classes
        features, output_dim, auxillery_layers = self._make_layers(self.cfg, input_ch, input_dim, feat_mult)

        for layer in self.cfg:
            if isinstance(layer, int):
                output_ch = layer
        if args.num_layers > 0: 
            classifier = Net(args.num_layers, args.num_hidden, output_dim, int(output_ch * feat_mult), num_classes)
            features.extend([*classifier.layers])
            auxillery_layers.extend([*classifier.auxillery_layers])
        else:
            classifier = nn.Linear(output_dim*output_dim*int(output_ch * feat_mult), num_classes)
            features.append(classifier)
            auxillery_layers.append(nn.Identity(1))

        #self.main_cnn = rep(nn.ModuleList(features))
        self.features = nn.ModuleList(features)
        #self.auxillary_nets = nn.ModuleList(auxillery_layers)
        self.auxillery_layers = nn.ModuleList(auxillery_layers)

            
    def parameters(self):
        if not args.backprop:
            return self.features[len(self.features) - 1].parameters()
        else:
            return super(VGGn, self).parameters()
    
    def _make_layers(self, cfg, input_ch, input_dim, feat_mult):
        layers = []
        auxillery_layers = []
        first_layer = True
        scale_cum = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                auxillery_layers += [nn.Identity(1)]
                scale_cum *=2
            elif x == 'M3':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
                auxillery_layers += [nn.Identity()]
                scale_cum *=2
            elif x == 'M4':
                layers += [nn.MaxPool2d(kernel_size=4, stride=4)]
                auxillery_layers += [nn.Identity()]
                scale_cum *=4
            elif x == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                auxillery_layers += [nn.Identity()]
                scale_cum *=2
            elif x == 'A3':
                layers += [nn.AvgPool2d(kernel_size=3, stride=2, padding=1)]
                auxillery_layers += [nn.Identity()]
                scale_cum *=2
            elif x == 'A4':
                layers += [nn.AvgPool2d(kernel_size=4, stride=4)]
                auxillery_layers += [nn.Identity()]
                scale_cum *=4
            else:
                x = int(x * feat_mult)
                if first_layer and input_dim > 64:
                    scale_cum = 2
                    layers += [LocalLossBlockConv(input_ch, x, kernel_size=7, stride=2, padding=3, 
                                             num_classes=self.num_classes, 
                                             dim_out=input_dim//scale_cum, 
                                             first_layer=first_layer)]
                    print(scale_cum, input_dim, self.num_classes, x)
                    auxillery_layers += [Auxillery("conv", input_dim//scale_cum, self.num_classes, x)]
                else:
                    layers += [LocalLossBlockConv(input_ch, x, kernel_size=3, stride=1, padding=1, 
                                             num_classes=self.num_classes, 
                                             dim_out=input_dim//scale_cum, 
                                             first_layer=first_layer)]
                    print(scale_cum, input_dim, self.num_classes, x)
                    print(input_dim//scale_cum)
                    auxillery_layers += [Auxillery("conv", input_dim//scale_cum, self.num_classes, x)]

                input_ch = x
                first_layer = False
        
        # maybe make module list 
        return nn.ModuleList(layers), input_dim//scale_cum, nn.ModuleList(auxillery_layers)


