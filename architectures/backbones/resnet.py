import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import export_fn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import ResNet as BaseResNet


class ResNet(BaseResNet):
    def __init__(self, block, layers, zero_init_residual = True, groups = 1, 
                 width_per_group = 64, replace_stride_with_dilation = None, 
                 norm_layer = None, teacher=False, *args, **kwargs):
        super(BaseResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_shape = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def forward(self, x, return_fmaps=None):
        if return_fmaps is not None and return_fmaps == "last":
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return x
        elif return_fmaps is not None and return_fmaps == "all":
            fmaps = [self.layer0(x)]
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                fmaps.append(layer(fmaps[-1]))
            return fmaps[1:]
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        
    def set_eval_state(self):
        return None

@export_fn
def resnet18(args, teacher=False):
    resnet = ResNet(BasicBlock, [2, 2, 2, 2], teacher=teacher, **args.__dict__)
    return resnet


@export_fn
def resnet34(args, teacher=False):
    resnet = ResNet(BasicBlock, [3, 4, 6, 3], teacher=teacher, **args.__dict__)
    return resnet

@export_fn
def resnet50(args, teacher=False):
    resnet = ResNet(Bottleneck, [3, 4, 6, 3], teacher=teacher, **args.__dict__)
    return resnet


@export_fn
def resnet50w2(args, teacher=False):
    resnet = ResNet(BasicBlock, [3, 4, 6, 3], teacher=teacher, widen=2, **args.__dict__)
    return resnet


@export_fn
def resnet50w4(args, teacher=False):
    resnet = ResNet(BasicBlock, [3, 4, 6, 3], teacher=teacher, widen=4, **args.__dict__)
    return resnet


@export_fn
def resnet50w5(args, teacher=False):
    resnet = ResNet(BasicBlock, [3, 4, 6, 3], teacher=teacher, widen=5, **args.__dict__)
    return resnet
