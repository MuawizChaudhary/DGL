import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import similarity_matrix


class rep(nn.Module):
    def __init__(self, blocks):
        super(rep, self).__init__()
        self.blocks = blocks

    def forward(self, x, n, upto=False):
        # if upto = True we forward from the input to output of layer n
        # if upto = False we forward just through layer n
        if upto:
            for i in range(n + 1):
                if isinstance(self.blocks[i], nn.MaxPool2d) or isinstance(self.blocks[i], nn.Linear):
                    out = self.blocks[i](x)
                    return out, out
                elif isinstance(self.blocks[i], LocalLossBlockLinear):
                    x = x.view(x.size(0), -1)
                    x, x_return = self.forward(x, i, upto=False)
                    return x, x_return
                else:
                    x, x_return = self.forward(x, i, upto=False)
                    return x, x_return

        if isinstance(self.blocks[n], nn.MaxPool2d) or isinstance(self.blocks[n], nn.Linear):
            out = self.blocks[n](x)
            return out, out
        elif isinstance(self.blocks[n], LocalLossBlockLinear):
            x = x.view(x.size(0), -1)
            out, out_return = self.blocks[n](x)
            return out, out_return
        else:
            out, out_return = self.blocks[n](x)
            return out, out_return


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

    def __init__(self, num_in, num_out, num_classes, dropout=0.0,
                 nonlin="relu", first_layer=False, bn=False):
        super(LocalLossBlockLinear, self).__init__()
        self.num_classes = num_classes
        self.first_layer = first_layer
        self.dropout_p = dropout

        encoder = nn.Linear(num_in, num_out, bias=True)

        if bn:
            batchnorm = torch.nn.BatchNorm1d(num_out)
            nn.init.constant_(batchnorm.weight, 1)
            nn.init.constant_(batchnorm.bias, 0)
        else:
            batchnorm = nn.Identity()

        if nonlin == 'relu':
            nonlin = nn.ReLU(inplace=True)
        elif nonlin == 'leakyrelu':
            nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.MLP = nn.Sequential(encoder, batchnorm, nonlin)

        if self.dropout_p > 0:
            self.dropout = torch.nn.Dropout(p=self.dropout_p, inplace=False)

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

    def __init__(self, ch_in, ch_out, kernel_size, stride, padding,
                 num_classes, dim_out, dropout=0.0, nonlin="relu",
                 first_layer=False, bn=False):
        super(LocalLossBlockConv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.num_classes = num_classes
        self.first_layer = first_layer
        self.dropout_p = dropout

        encoder = nn.Conv2d(ch_in, ch_out, kernel_size, stride=stride, padding=padding)

        if bn:
            batchnorm = torch.nn.BatchNorm2d(ch_out)
            nn.init.constant_(batchnorm.weight, 1)
            nn.init.constant_(batchnorm.bias, 0)
        else:
            batchnorm = nn.Identity()

        if nonlin == 'relu':
            nonlin = nn.ReLU(inplace=True)
        elif nonlin == 'leakyrelu':
            nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.MLP = nn.Sequential(encoder, batchnorm, nonlin)

        if self.dropout_p > 0:
            self.dropout = torch.nn.Dropout2d(p=self.dropout_p, inplace=False)

    def forward(self, x):
        h = self.MLP(x)
        h_return = h
        if self.dropout_p > 0:
            h_return = self.dropout(h_return)
        return h, h_return


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

    def __init__(self, num_layers, num_hidden, input_dim, input_ch,
                 num_classes, no_similarity_std, dropout=0.0, nonlin='relu',
                 loss_sup="predsim", dim_in_decoder=2048, 
                 aux_type="nokland", n_mlp=0, bn=False,
                 aux_bn=False):
        super(Net, self).__init__()

        self.num_hidden = num_hidden
        self.num_layers = num_layers
        reduce_factor = 1

        self.layers = nn.ModuleList([LocalLossBlockLinear(input_dim * input_dim * input_ch,
                                                          num_hidden, num_classes, dropout=dropout, nonlin=nonlin,
                                                          first_layer=True,
                                                          bn=bn)])

        self.auxillery_layers = nn.ModuleList([auxillary_linear_classifier(num_hidden,
                num_classes=num_classes,
                n_mlp=n_mlp, 
                loss_sup=loss_sup, bn=aux_bn,
                dropout=dropout)])

        for i in range(1, num_layers):
            layer = LocalLossBlockLinear(int(num_hidden // (reduce_factor ** (i - 1))),
                                         int(num_hidden // (reduce_factor ** i)),
                                         num_classes, dropout=dropout, nonlin=nonlin)
            self.layers.extend([layer])

        #TODO fix
        for i in range(1, num_layers):
            aux = Auxillery("linear", int(num_hidden // (reduce_factor ** i)),
                            num_classes, int(num_hidden // (reduce_factor ** i)),
                            no_similarity_std, loss_sup, dim_in_decoder)
            self.auxillery_layers.extend([aux])

        layer_out = nn.Linear(int(num_hidden // (reduce_factor ** (num_layers - 1))), num_classes)
        layer_out.weight.data.zero_()

        self.layers.extend([layer_out])
        self.auxillery_layers.extend([nn.Identity()])

    def parameters(self):
        return self.layer_out.parameters()


cfg = {
    'vgg6a': [128, 'M', 256, 'M', 512, 'M', 512],
    'vgg6b': [128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'vgg8': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'vgg8a': [128, 256, 'M', 256, 512, 'M', 512, 'M', 512],
    'vgg8b': [128, 256, 'M', 256, 512, 'M', 512, 'M', 512, 'M'],
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg11b': [128, 128, 128, 256, 'M', 256, 512, 'M', 512, 512, 'M', 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
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

    def __init__(self, vggname, input_dim=32, input_ch=3, num_classes=10,
                 feat_mult=1, dropout=0.0, nonlin="relu", no_similarity_std=False,
                 loss_sup="predsim", dim_in_decoder=2048,
                 num_layers=0, num_hidden=1024,
                 aux_type="nokland", n_mlp=0, n_conv=0, 
                 pooling="avg", bn=True, aux_bn=False):
        super(VGGn, self).__init__()
        self.cfg = cfg[vggname]
        self.input_dim = input_dim
        self.input_ch = input_ch
        self.num_classes = num_classes
        self.no_similarity_std = no_similarity_std
        self.dropout = dropout
        self.nonlin = nonlin
        self.loss_sup = loss_sup
        self.dim_in_decoder = dim_in_decoder
        self.aux_type = aux_type
        self.n_mlp = n_mlp
        self.n_conv = n_conv
        self.pooling = pooling
        self.bn = bn
        self.aux_bn = aux_bn

        features, auxillery_layers, output_dim = self._make_layers(self.cfg, input_ch, input_dim, feat_mult)

        for layer in self.cfg:
            if isinstance(layer, int):
                output_ch = layer
        if num_layers > 0:
            classifier = Net(num_layers, num_hidden, output_dim,
                             int(output_ch * feat_mult), num_classes,
                             self.no_similarity_std, self.dropout, nonlin=nonlin,
                             loss_sup=loss_sup, dim_in_decoder=dim_in_decoder,
                             aux_type=aux_type, n_mlp=n_mlp,
                             bn=bn, aux_bn=aux_bn)
            features.extend([*classifier.layers])
            auxillery_layers.extend([*classifier.auxillery_layers])
        else:
            classifier = nn.Linear(output_dim * output_dim * int(output_ch * feat_mult), num_classes)
            features.append(classifier)
            auxillery_layers.append(nn.Identity())

        self.main_cnn = rep(nn.ModuleList(features))
        self.auxillary_nets = nn.ModuleList(auxillery_layers)

    def forward(self, representation, n, upto=False):
        rep, rep_return = self.main_cnn.forward(representation, n, upto=upto)
        outputs = self.auxillary_nets[n](rep)
        return outputs, rep_return

    def _make_layers(self, cfg, input_ch, input_dim, feat_mult):
        layers = []
        auxillery_layers = []
        first_layer = True
        scale_cum = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                auxillery_layers += [nn.Identity()]
                scale_cum *= 2
            else:
                x = int(x * feat_mult)
                if first_layer and input_dim > 64:
                    scale_cum = 2
                    layers += [LocalLossBlockConv(input_ch, x, kernel_size=7, stride=2, padding=3,
                                                  num_classes=self.num_classes,
                                                  dim_out=input_dim // scale_cum,
                                                  dropout=self.dropout,
                                                  nonlin=self.nonlin,
                                                  first_layer=first_layer,
                                                  bn=self.bn)]

                    auxillery_layers += [auxillary_conv_classifier(x, input_dim // scale_cum,
                                cnn=False, num_classes=self.num_classes,
                                n_mlp=self.n_mlp,
                                n_conv=self.n_conv, 
                                loss_sup=self.loss_sup,
                                dim_in_decoder_arg=self.dim_in_decoder,
                                pooling=self.pooling, bn=self.aux_bn, 
                                dropout=self.dropout)]
                else:
                    layers += [LocalLossBlockConv(input_ch, x, kernel_size=3, stride=1, padding=1,
                                                  num_classes=self.num_classes,
                                                  dim_out=input_dim // scale_cum,
                                                  dropout=self.dropout,
                                                  nonlin=self.nonlin,
                                                  first_layer=first_layer,
                                                  bn=self.bn)]

                    auxillery_layers += [auxillary_conv_classifier(x, input_dim // scale_cum,
                                cnn=False, num_classes=self.num_classes,
                                n_mlp=self.n_mlp,
                                n_conv=self.n_conv, 
                                loss_sup=self.loss_sup,
                                dim_in_decoder_arg = self.dim_in_decoder,
                                pooling=self.pooling, bn=self.aux_bn,
                                dropout=self.dropout)]
                input_ch = x
                first_layer = False

        return nn.ModuleList(layers), nn.ModuleList(auxillery_layers), input_dim // scale_cum



class auxillary_conv_classifier(nn.Module):
    def __init__(self, input_features=256, in_size=32, cnn=False,
                 num_classes=10, n_mlp=0, n_conv=0, loss_sup="pred", 
                 dim_in_decoder_arg=2048, pooling="avg",
                 bn=False, dropout=0.0):
        super(auxillary_conv_classifier, self).__init__()
        self.in_size = in_size
        self.cnn = cnn
        feature_size = input_features
        self.loss_sup = loss_sup
        input_features = in_size
        in_size = feature_size
        self.dim_in_decoder = dim_in_decoder_arg
        self.pooling = pooling
        self.pool = nn.Identity()
        self.blocks = []
        
        for n in range(n_conv):
            if bn:
                bn_temp = nn.BatchNorm2d(feature_size)
            else:
                bn_temp = nn.Identity()

            relu_temp = nn.ReLU(True)

            conv = nn.Conv2d(feature_size, feature_size,
                                 kernel_size=1, stride=1, padding=0, bias=False)

            self.blocks.append(nn.Sequential(conv, bn_temp, relu_temp))
        self.blocks = nn.ModuleList(self.blocks)

        if (loss_sup == 'pred' or loss_sup == 'predsim') and pooling == "avg":
            # Resolve average-pooling kernel size in order for flattened dim to match args.dim_in_decoder
            ks_h, ks_w = 1, 1
            dim_out_h, dim_out_w = input_features, input_features
            self.dim_in_decoder = in_size * dim_out_h * dim_out_w
            while self.dim_in_decoder > dim_in_decoder_arg and ks_h < input_features:
                ks_h *= 2
                dim_out_h = math.ceil(input_features / ks_h)
                self.dim_in_decoder = in_size * dim_out_h * dim_out_w
                if self.dim_in_decoder > dim_in_decoder_arg:
                    ks_w *= 2
                    dim_out_w = math.ceil(input_features / ks_w)
                    self.dim_in_decoder = in_size * dim_out_h * dim_out_w
            if ks_h > 1 or ks_w > 1:
                pad_h = (ks_h * (dim_out_h - input_features // ks_h)) // 2
                pad_w = (ks_w * (dim_out_w - input_features // ks_w)) // 2
                self.pool = nn.AvgPool2d((ks_h, ks_w), padding=(pad_h, pad_w))
                self.bn = nn.Identity()
            else:
                self.pool = nn.Identity()
                self.bn = nn.Identity()

        #dropout=0.0


        if pooling == "adaptiveavg":
            self.dim_in_decoder = feature_size*4
            self.pool = nn.AdaptiveAvgPool2d((2, 2))

        if not bn:
            self.bn = nn.Identity()
        else:
            self.bn = nn.BatchNorm2d(feature_size)

        if n_mlp > 0:
            mlp_feat = self.dim_in_decoder

            layers = []

            for l in range(n_mlp):
                if bn:
                    bn_temp = nn.BatchNorm1d(mlp_feat)
                else:
                    bn_temp = nn.Identity()
                dropout_temp = torch.nn.Dropout(p=dropout, inplace=False)
                layers += [nn.Linear(mlp_feat, mlp_feat),
                           bn_temp, nn.ReLU(True), dropout_temp]
            self.mlp = True
            self.preclassifier = nn.Sequential(*layers)
            self.classifier = nn.Linear(mlp_feat, num_classes)
            self.classifier.weight.data.zero_()
            if loss_sup == 'predsim':
                self.sim_loss = nn.Conv2d(feature_size, feature_size, 3, stride=1, padding=1, bias=False)

        else:
            self.mlp = False
            self.preclassifier = nn.Identity()
            self.classifier = nn.Linear(self.dim_in_decoder, num_classes)
            self.classifier.weight.data.zero_()
            if loss_sup == 'predsim':
                self.sim_loss = nn.Conv2d(feature_size, feature_size, 3, stride=1, padding=1, bias=False)
    
    def forward(self, x):
        out = None
        loss_sim = None
        if self.loss_sup == "predsim":
            loss_sim = self.sim_loss(x)

        if self.pooling == 'adapativeavg':
            x = F.adaptive_avg_pool2d(x, (math.ceil(self.in_size / 4), math.ceil(self.in_size / 4)))
        for block in self.blocks:
            x = block(x)
        out = self.pool(x)
        if not self.mlp:
            out = self.bn(out)
        out = out.view(out.size(0), -1)
        out = self.preclassifier(out)
        out = self.classifier(out)
        return (loss_sim, out)


class auxillary_linear_classifier(nn.Module):
    def __init__(self, input_features=256,
                 num_classes=10, n_mlp=0, 
                 loss_sup="pred", bn=False, dropout=0.0):
        super(auxillary_linear_classifier, self).__init__()
        feature_size = input_features
        self.loss_sup = loss_sup
        #dropout=0.0

        if not bn:
            self.bn = nn.Identity()
        else:
            self.bn = nn.BatchNorm2d(feature_size)


        if n_mlp > 0:
            self.mlp = True
            mlp_feat = feature_size
            layers = []

            for l in range(n_mlp):
                in_feat = mlp_feat
                if bn:
                    bn_temp = nn.BatchNorm1d(mlp_feat)
                else:
                    bn_temp = nn.Identity()
                dropout_temp = torch.nn.Dropout(p=dropout, inplace=False)
                layers += [nn.Linear(in_feat, mlp_feat),
                           bn_temp, nn.ReLU(True), dropout_temp]

            self.preclassifier = nn.Sequential(*layers)
            self.classifier = nn.Linear(mlp_feat, num_classes)
            self.classifier.weight.data.zero_()

            if loss_sup == 'predsim':
                self.sim_loss = nn.Linear(feature_size, feature_size, bias=False)
        else:
            self.mlp = False
            self.preclassifier = nn.Identity()
            self.classifier = nn.Linear(feature_size, num_classes)
            self.classifier.weight.data.zero_()

            if loss_sup == 'predsim':
                self.sim_loss = nn.Linear(feature_size, feature_size, bias=False)


    def forward(self, x):
        out = None
        loss_sim = None

        if self.loss_sup == "predsim":
            loss_sim = self.sim_loss(x)
        if not self.mlp:
            x = self.bn(x)

        out = x.view(x.size(0), -1)
        out = self.preclassifier(out)
        out = self.classifier(out)
        return (loss_sim, out)
