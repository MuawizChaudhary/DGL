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
    def __init__(self, depth=4, num_classes=10,
                 feature_size=64, downsample=None):
        super(Net, self).__init__()

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
            auxillary_classifier2(input_features=feature_size,
                                  in_size=self.in_size, num_classes=num_classes,
                                  n_lin=3, mlp_layers=3))
        
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
                    auxillary_classifier2(input_features=feature_size,
                                          in_size=self.in_size, num_classes=num_classes,
                                          n_lin=3,mlp_layers=3))

        self.auxillary_nets.append(auxillary_classifier2(input_features=feature_size,
                                          in_size=self.in_size, num_classes=num_classes,
                                          n_lin=0,mlp_layers=2))
        self.main_cnn = rep(self.blocks)
        
    def forward(self, representation, n, upto=False):
        representation = self.main_cnn.forward(representation, n, upto=upto)
        outputs = self.auxillary_nets[n](representation)
        return outputs, representation



class auxillary_classifier2(nn.Module):
    def __init__(self, input_features=256, in_size=32,
                 num_classes=10, n_lin=0, mlp_layers=0):
        super(auxillary_classifier2, self).__init__()
        self.n_lin=n_lin
        self.in_size=in_size

        feature_size = input_features
        self.blocks = []
        for n in range(self.n_lin):
            if n==0:
                input_features = input_features
            else:
                input_features = feature_size

            bn_temp = nn.BatchNorm2d(feature_size)


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
            self.classifier = nn.Linear(feature_size*4, num_classes)


    def forward(self, x):
        out = x
        #First reduce the size by 16
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

