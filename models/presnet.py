from __future__ import absolute_import
import math

import torch.nn as nn

import numpy as np

import torch

from utils.common_utils import try_cuda
from utils.prune_utils import (ConvLayerRotation,
                               LinearLayerRotation,
                               register_bottleneck_layer,
                               update_QQ_dict)
from utils.common_utils import try_cuda

__all__ = ['presnet', 'BottleneckPResNet']

"""
preactivation resnet with bottleneck design.
"""


class channel_selection(nn.Module):
    """
    Select channels from the output of BatchNorm2d layer. It should be put directly after BatchNorm2d layer.
    The output shape of this layer is determined by the number of 1 in `self.indexes`.
    """
    def __init__(self, num_channels):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
	    """
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))

    def forward(self, input_tensor):
        """
        Parameter
        ---------
        input_tensor: (N,C,H,W). It should be the output of BatchNorm2d layer.
		"""
        selected_index = np.squeeze(np.argwhere(self.indexes.data.cpu().numpy()))
        if selected_index.size == 1:
            selected_index = np.resize(selected_index, (1,))
        output = input_tensor[:, selected_index, :, :]
        return output


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        is_pruned = hasattr(self.conv1, 'in_indices')
        if is_pruned:
            indices = []
            indices.append(self.conv3.out_indices)

        residual = x

        out = self.bn1(x)
        # out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            if is_pruned:
                indices.append(self.downsample[0].out_indices)
        elif is_pruned:
            indices.append(self.conv1.in_indices)

        if is_pruned:
            n_c = len(set(indices[0] + indices[1]))
            all_indices = list(set(indices[0] + indices[1]))
            r_indices = []
            o_indices = []

            for i in range(n_c):
                idx = all_indices[i]
                if idx in indices[0] and idx in indices[1]:
                    r_indices.append(i)
                    o_indices.append(i)
                elif idx in indices[0]:
                    o_indices.append(i)
                elif idx in indices[1]:
                    r_indices.append(i)
            res = try_cuda(torch.zeros(x.size(0), n_c, residual.size(2), residual.size(3)))
            res[:, r_indices, :, :] = residual
            res[:, o_indices, :, :] += out
            out = res
        else:
            out += residual

        return out


class presnet(nn.Module):
    def __init__(self, depth=164, dataset='cifar10', cfg=None):
        super(presnet, self).__init__()
        assert (depth - 2) % 9 == 0, 'depth should be 9n+2'

        n = (depth - 2) // 9
        block = Bottleneck

        if cfg is None:
            # Construct config variable.
            cfg = [[64, 64, 64], [64, 64, 64]*(n-1), [64, 64, 64], [128, 128, 128]*(n-1), [128, 128, 128], [256, 256, 256]*(n-1), [256]]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 64, n, cfg = cfg[0:3*n])
        self.layer2 = self._make_layer(block, 128, n, cfg = cfg[3*n:6*n], stride=2)
        self.layer3 = self._make_layer(block, 256, n, cfg = cfg[6*n:9*n], stride=2)
        self.bn = nn.BatchNorm2d(256 * block.expansion)
        self.select = channel_selection(256 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        if dataset == 'cifar10':
            self.fc = nn.Linear(cfg[-1], 10)
        elif dataset == 'cifar100':
            self.fc = nn.Linear(cfg[-1], 100)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0:3], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[3*i: 3*(i+1)]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        # x = self.select(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class BottleneckPResNet(nn.Module):
    def __init__(self, net_prev, fix_rotation=True):
        super(BottleneckPResNet, self).__init__()
        self.conv1 = net_prev.conv1

        self.layer1 = net_prev.layer1
        self.layer2 = net_prev.layer2
        self.layer3 = net_prev.layer3
        self.bn = net_prev.bn
        self.fc = net_prev.fc
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fix_rotation = fix_rotation
        self._is_registered = False

    def _update_bottleneck(self, bneck, modules, Q_g, Q_a, W_star, use_patch, fix_rotation):
        m = bneck.conv1
        if isinstance(m, nn.Sequential):
            m = m[1]
        if m in modules:
            bneck.conv1 = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
            update_QQ_dict(Q_g, Q_a, m, bneck.conv1[1])

        m = bneck.conv2
        if isinstance(m, nn.Sequential):
            m = m[1]
        if m in modules:
            bneck.conv2 = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
            update_QQ_dict(Q_g, Q_a, m, bneck.conv2[1])

        m = bneck.conv3
        if isinstance(m, nn.Sequential):
            m = m[1]
        if m in modules:
            bneck.conv3 = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
            update_QQ_dict(Q_g, Q_a, m, bneck.conv3[1])

        m = bneck.downsample
        if m is not None:
            if len(m) == 1 and m[0] in modules:
                m = m[0]
                bneck.downsample = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
                update_QQ_dict(Q_g, Q_a, m, bneck.downsample[1])
            elif len(m) == 3 and m[1] in modules:
                m = m[1]
                bneck.downsample = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
                update_QQ_dict(Q_g, Q_a, m, bneck.downsample[1])
            else:
                assert len(m) == 1 or len(m) == 3, 'Upexpected layer %s' % m

    def register(self, modules, Q_g, Q_a, W_star, use_patch, fix_rotation, re_init):
        for m in self.modules():
            if isinstance(m, Bottleneck):
                self._update_bottleneck(m, modules, Q_g, Q_a, W_star, use_patch, fix_rotation)

        m = self.conv1
        if isinstance(m, nn.Sequential):
            m = m[1]
        if m in modules:
            self.conv1 = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
            update_QQ_dict(Q_g, Q_a, m, self.conv1[1])

        m = self.fc
        if isinstance(m, nn.Sequential):
            m = m[1]
        if m in modules:
            self.fc = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
            update_QQ_dict(Q_g, Q_a, m, self.fc[1])
        self._is_registered = True
        if re_init:
            raise NotImplementedError
            # self.apply(_weights_init)

    def forward(self, x):
        assert self._is_registered
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        # x = self.select(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x