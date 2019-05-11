import math
import torch
import torch.nn as nn

from utils.common_utils import try_contiguous
from utils.prune_utils import register_bottleneck_layer, update_QQ_dict
from utils.prune_utils import LinearLayerRotation, ConvLayerRotation
# from layers.bottleneck_layers import LinearBottleneck, Conv2dBottleneck
from models.resnet import _weights_init

_AFFINE = True
# _AFFINE = False

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None):
        super(VGG, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.feature = self.make_layers(cfg, True)
        self.dataset = dataset
        if dataset == 'cifar10' or dataset == 'cinic-10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'tiny_imagenet':
            num_classes = 200
        self.classifier = nn.Linear(cfg[-1], num_classes)
        if init_weights:
            self.apply(_weights_init)
        #    self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, affine=_AFFINE), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        if self.dataset == 'tiny_imagenet':
            x = nn.AvgPool2d(4)(x)
        else:
            x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1.0)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class BottleneckVGG(nn.Module):
    def __init__(self, vgg_prev, fix_rotation=True):
        super(BottleneckVGG, self).__init__()
        self.dataset = vgg_prev.dataset
        self.feature = vgg_prev.feature
        self.classifier = vgg_prev.classifier
        self.fix_rotation = fix_rotation
        self._is_registered = False

    def register(self, modules, Q_g, Q_a, W_star, use_patch, fix_rotation, re_init):
        n_seqs = len(self.feature)
        for idx in range(n_seqs):
            m = self.feature[idx]
            if isinstance(m, nn.Sequential):
                m = m[1]
            if m in modules:
                self.feature[idx] = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
                update_QQ_dict(Q_g, Q_a, m, self.feature[idx][1])
        m = self.classifier
        if isinstance(m, nn.Sequential):
            m = m[1]
        if m in modules:
            self.classifier = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
            update_QQ_dict(Q_g, Q_a, m, self.classifier)
        self._is_registered = True
        if re_init:
            self.apply(_weights_init)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1.0)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, LinearLayerRotation):
                if m.trainable:
                    print('* init Linear rotation')
                    m.rotation_matrix.data.normal_(0, 0.01)
            elif isinstance(m, ConvLayerRotation):
                if m.trainable:
                    print('* init Conv rotation')
                    n = 1 * m.rotation_matrix.size(1)
                    m.rotation_matrix.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        assert self._is_registered
        nseq = len(self.feature)
        for idx in range(nseq):
            x = self.feature[idx](x)

        if self.dataset == 'tiny_imagenet':
            x = nn.AvgPool2d(4)(x)
        else:
            x = nn.AvgPool2d(2)(x)
        out = x.view(x.size(0), -1)
        out = self.classifier(out)
        return out

