import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn.init import trunc_normal_

class L2Normalizer(nn.Module):
    def forward(self, x):
        return F.normalize(x,p=2,dim=-1)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, hidden_layers=1, activation=nn.ReLU, bn=True, l2norm=True):
        super(MLP, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        
        layers = []
        inp=input_dim
        for i in range(hidden_layers):
            layers.append(nn.Linear(inp, hidden_dim))
            if bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation())
            inp=hidden_dim
        layers.append(nn.Linear(inp, output_dim))
        if l2norm:
            layers.append(L2Normalizer())
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)

class CentroidLayer(nn.Linear):
    def forward(self, input, with_grads=True):
        input = F.normalize(input,p=2,dim=-1)
        if with_grads:
            return F.linear(input, self.weight, self.bias)
        else:
            return F.linear(input, self.weight.clone().detach(), self.bias)


class MLPBottleneckClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, hidden_layers=1, bottleneck_dim=256, 
                 activation=nn.ReLU, bn=True, predictor=False, predictor_grads=False,
                 local_crop_grads=True, predictor_activation=None):
        super(MLPBottleneckClassifier, self).__init__()
        self.projector = MLP(input_dim, bottleneck_dim, hidden_dim, hidden_layers, activation, bn, False)
        if predictor:
            self.predictor = MLP(bottleneck_dim, bottleneck_dim, hidden_dim, 1, activation, bn, False)
        else:
            self.predictor=None

        self.centroid_layer = weight_norm(CentroidLayer(bottleneck_dim, output_dim, bias=False))
        self.centroid_layer.weight_g.data.fill_(1)
        self.centroid_layer.weight_g.requires_grad = False

        self.predictor_grads=predictor_grads
        self.local_crop_grads = local_crop_grads
        if predictor_activation == "bn":
            self.predictor_activation = nn.BatchNorm1d(bottleneck_dim)
        elif predictor_activation == "ln":
            self.predictor_activation = nn.LayerNorm(bottleneck_dim)
        else:
            self.predictor_activation = nn.Identity()

    def forward(self, x, get_predictor=False):
        mc=False
        if isinstance(x, list):
            if len(x)==2:
                mc=True
                v_g = x[0].shape[0]
                v_l = x[1].shape[0]
            else:
                mc=False
            x = torch.cat(x,dim=0)
        projections = self.projector(x)
        if mc and not self.local_crop_grads:
            similarities_global = self.centroid_layer(projections[:v_g])
            similarities_local = self.centroid_layer(projections[v_g:], with_grads=False)
            similarities = torch.cat([similarities_global, similarities_local],dim=0)
        else:
            similarities = self.centroid_layer(projections)
        if self.predictor is not None and get_predictor:
            projections = self.predictor_activation(projections)
            predictions = self.predictor(projections)
            similarities = torch.cat([similarities, self.centroid_layer(predictions, with_grads=self.predictor_grads)],dim=0)
        return similarities, F.normalize(projections, p=2,dim=-1)
