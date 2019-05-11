import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.modules.utils import _pair


# ======================================================
# Find layer dependency
# Update input indices (adapt to previous layers)
# Update output indices
# ======================================================
def update_resnet_block_dependencies(prev_modules, block, dependencies):
    for m in prev_modules:
        assert isinstance(m, (nn.Conv2d, nn.Linear)), 'Only conv or linear layer can be previous modules.'
    dependencies[block.conv1] = prev_modules
    dependencies[block.bn1] = [block.conv1]
    dependencies[block.conv2] = [block.conv1]
    dependencies[block.bn2] = [block.conv2]

    if block.downsample is not None:
        dependencies[block.downsample[0]] = prev_modules
        dependencies[block.bn3] = [block.downsample[0]]


def update_resnet_layer_dependencies(prev_modules, layer, dependencies):
    num_blocks = len(layer)
    for block_idx in range(num_blocks):
        block = layer[block_idx]
        update_resnet_block_dependencies(prev_modules, block, dependencies)
        prev_modules = [block.conv2]
        if block.downsample is not None:
            prev_modules.append(block.downsample[0])
        else:
            prev_modules.extend(dependencies[block.conv1])


def update_presnet_block_dependencies(prev_modules, block, dependencies):
    # TODO: presnet
    for m in prev_modules:
        assert isinstance(m, (nn.Conv2d, nn.Linear)), 'Only conv or linear layer can be previous modules.'
    dependencies[block.bn1] = prev_modules
    dependencies[block.conv1] = prev_modules
    dependencies[block.bn2] = [block.conv1]
    dependencies[block.conv2] = [block.conv1]
    dependencies[block.bn3] = [block.conv2]
    dependencies[block.conv3] = [block.conv2]

    if block.downsample is not None:
        dependencies[block.downsample[0]] = prev_modules


def update_presnet_layer_dependencies(prev_modules, layer, dependencies):
    # TODO: presnet
    num_blocks = len(layer)
    for block_idx in range(num_blocks):
        block = layer[block_idx]
        update_presnet_block_dependencies(prev_modules, block, dependencies)
        prev_modules = [block.conv3]
        if block.downsample is not None:
            prev_modules.append(block.downsample[0])
        else:
            prev_modules.extend(dependencies[block.bn1])


def get_layer_dependencies(model, network):
    # Helper function; ad-hoc fix
    dependencies = OrderedDict()
    if 'vgg' in network:
        modules = model.modules()
        prev_layers = []
        for m in modules:
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                dependencies[m] = prev_layers
                prev_layers = [m]
            elif isinstance(m, nn.BatchNorm2d):
                dependencies[m] = prev_layers
    elif 'presnet' in network:
        dependencies[model.conv1] = []
        prev_modules = [model.conv1]

        # update first layer's dependencies
        update_presnet_layer_dependencies(prev_modules, model.layer1, dependencies)

        # update second layer's dependencies
        prev_modules = [model.layer1[-1].conv3]
        if model.layer1[-1].downsample is not None:
            prev_modules.append(model.layer1[-1].downsample[0])
        else:
            prev_modules = [model.layer1[-1].conv3] + dependencies[model.layer1[-1].bn1]
        update_presnet_layer_dependencies(prev_modules, model.layer2, dependencies)

        # update third layer's dependencies
        prev_modules = [model.layer2[-1].conv3]
        if model.layer2[-1].downsample is not None:
            prev_modules.append(model.layer2[-1].downsample[0])
        else:
            prev_modules = [model.layer2[-1].conv3] + dependencies[model.layer2[-1].bn1]
        update_presnet_layer_dependencies(prev_modules, model.layer3, dependencies)

        # update bn and fc layer's dependencies
        prev_modules = [model.layer3[-1].conv3]
        if model.layer3[-1].downsample is not None:
            prev_modules.append(model.layer3[-1].downsample[0])
        else:
            prev_modules = [model.layer3[-1].conv3] + dependencies[model.layer3[-1].bn1]
        dependencies[model.bn] = prev_modules
        dependencies[model.fc] = prev_modules

    elif 'resnet' in network:
        dependencies[model.conv1] = []
        dependencies[model.bn] = [model.conv1]

        prev_modules = [model.conv1]
        update_resnet_layer_dependencies(prev_modules, model.layer1, dependencies)

        prev_modules = [model.layer1[-1].conv2]
        if model.layer1[-1].downsample is not None:
            prev_modules.append(model.layer1[-1].downsample[0])
        else:
            prev_modules = [model.layer1[-1].conv2] + dependencies[model.layer1[-1].conv1]
        update_resnet_layer_dependencies(prev_modules, model.layer2, dependencies)

        prev_modules = [model.layer2[-1].conv2]
        if model.layer2[-1].downsample is not None:
            prev_modules.append(model.layer2[-1].downsample[0])
        else:
            prev_modules = [model.layer2[-1].conv2] + dependencies[model.layer2[-1].conv1]
        update_resnet_layer_dependencies(prev_modules, model.layer3, dependencies)

        prev_modules = [model.layer3[-1].conv2]
        if model.layer3[-1].downsample is not None:
            prev_modules.append(model.layer3[-1].downsample[0])
        else:
            prev_modules = [model.layer3[-1].conv2] + dependencies[model.layer3[-1].conv1]
        dependencies[model.linear] = prev_modules

    return dependencies


def update_indices(model, network):
    dependencies = get_layer_dependencies(model, network)
    update_out_indices(model, dependencies)
    update_in_dinces(dependencies)


def update_out_indices(model, dependencies):
    pass


def update_in_dinces(dependencies):
    for m, deps in dependencies.items():
        if len(deps) > 0:
            indices = set()
            for d in deps:
                indices = indices.union(d.out_indices)
            m.in_indices = sorted(list(indices))

# def update_out_indices(model, dependencies):
#     closure = dict()
#     for _, deps in dependencies.items():
#         for m in deps:
#             closure[m] = closure.get(m, []) + deps
#
#     for m in model.modules():
#         if isinstance(m, (nn.Conv2d, nn.Linear)):
#             deps = dependencies[m]
#             if len(deps) >= 2:
#                 indices = set()
#                 for d in closure[deps[0]]:
#                     indices = indices.union(d.out_indices)
#                 indices = sorted(indices)
#                 for d in closure[deps[0]]:
#                     d.out_indices = list(indices)
#
#
# def update_in_dinces(dependencies):
#     for m, deps in dependencies.items():
#         if len(deps) > 0:
#             m.in_indices = deps[0].out_indices


# ======================================================
# For building vgg net: generate cfgs and generate mask
# as well as copying weights.
# ======================================================
def gen_network_cfgs(filter_nums, network):
    # [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
    if network == 'vgg19':
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
        counts = 0
        for idx in range(len(cfg)):
            c = cfg[idx]
            if c == 'M':
                counts += 1
                continue
            cfg[idx] = filter_nums[idx-counts]
    else:
        raise NotImplementedError
    return cfg


def copy_weights(m0, m1, ):
    if isinstance(m0, nn.BatchNorm2d):
        pass
    elif isinstance(m0, nn.Conv2d):
        pass
    elif isinstance(m0, nn.Linear):
        pass
    else:
        raise NotImplementedError


def get_threshold(values, percentage):
    v_sorted = sorted(values)
    n = int(len(values) * percentage)
    threshold = v_sorted[n]
    return threshold


def filter_indices(values, threshold):
    indices = []
    for idx, v in enumerate(values):
        if v > threshold:
            indices.append(idx)
    if len(indices) <= 1:
        # we make it at least 1 filters in each laer
        indices = [0]
    return indices


def get_rotation_layer_weights(model, qm):
    for m in model.modules():
        if (isinstance(m, nn.Sequential)
                and len(m) == 3
                and isinstance(m[0], (LinearLayerRotation, ConvLayerRotation))
                and isinstance(m[2], (LinearLayerRotation, ConvLayerRotation))):
            if qm is m[1]:
                return m[0].rotation_matrix.data, m[2].rotation_matrix.data
    raise ValueError('%s not found in the model. Potential bug!' % qm)


def update_QQ_dict(Q_g, Q_a, m, n):
    if n is not m:
        Q_g[n] = Q_g[m]
        Q_a[n] = Q_a[m]
        Q_a.pop(m)
        Q_g.pop(m)


def get_block_sum(m, imps):
    importances = []
    if isinstance(m, nn.Conv2d):
        kernel_size = m.kernel_size
        k = kernel_size[0] * kernel_size[1]
        l = imps.squeeze().size(0)
        bias = 1 if m.bias is not None else 0
        assert ((l-bias) // k) * k == (l-bias)
        for idx in range(0, l, k):
            s = min(idx+k, l)
            s = imps[idx:idx+k].sum().item()
            importances.append(s)
        return imps.new(importances)
    elif isinstance(m, nn.Linear):
        return imps


def count_module_params(m):
    counts = m.weight.view(-1).size(0)
    if m.bias is not None:
        counts += m.bias.size(0)
    return counts


class LinearLayerRotation(nn.Module):
    def __init__(self, rotation_matrix, bias=0, trainable=False):
        super(LinearLayerRotation, self).__init__()
        self.rotation_matrix = rotation_matrix
        self.rotation_matrix.requires_grad_(trainable)
        if trainable:
            self.rotation_matrix = nn.Parameter(self.rotation_matrix)
        
        self.trainable = trainable
        self.bias = bias

    def forward(self, x):
        if self.bias != 0:
            x = torch.cat([x, x.new(x.size(0), 1).fill_(self.bias)], 1)
        return x @ self.rotation_matrix

    def parameters(self):
        return [self.rotation_matrix]

    def extra_repr(self):
        return "in_features=%s, out_features=%s, trainable=%s" % (self.rotation_matrix.size(1),
                                                                  self.rotation_matrix.size(0),
                                                                  self.trainable)


class ConvLayerRotation(nn.Module):
    def __init__(self, rotation_matrix, bias=0, trainable=False):
        super(ConvLayerRotation, self).__init__()
        self.rotation_matrix = rotation_matrix.unsqueeze(2).unsqueeze(3)  # out_dim * in_dim
        self.rotation_matrix.requires_grad_(trainable)
        if trainable:
            self.rotation_matrix = nn.Parameter(self.rotation_matrix)
        self.trainable = trainable
        self.bias = bias

    def forward(self, x):
        # x: batch_size * in_dim * w * h
        if self.bias != 0:
            x = torch.cat([x, x.new(x.size(0), 1, x.size(2), x.size(3)).fill_(self.bias)], 1)
        return F.conv2d(x, self.rotation_matrix, None, _pair(1), _pair(0), _pair(1), 1)

    def parameters(self):
        return [self.rotation_matrix]

    def extra_repr(self):
        return "in_channels=%s, out_channels=%s, trainable=%s" % (self.rotation_matrix.size(1),
                                                                  self.rotation_matrix.size(0),
                                                                  self.trainable)


def register_bottleneck_layer(m, Q_g, Q_a, W_star, use_patch, trainable=False):
    assert use_patch
    if isinstance(m, nn.Linear):
        scale = nn.Linear(W_star.size(1), W_star.size(0), bias=False).cuda()
        scale.weight.data.copy_(W_star)
        bias = 1.0 if m.bias is not None else 0
        return nn.Sequential(
            LinearLayerRotation(Q_a, bias, trainable),
            scale,
            LinearLayerRotation(Q_g.t(), trainable=trainable))
    elif isinstance(m, nn.Conv2d):
        # if it is a conv layer, W_star should be out_c * in_c * h * w
        W_star = W_star.view(W_star.size(0), m.kernel_size[0], m.kernel_size[1], -1)
        W_star = W_star.transpose(2, 3).transpose(1, 2).contiguous()
        scale = nn.Conv2d(W_star.size(1), W_star.size(0), m.kernel_size,
                          m.stride, m.padding, m.dilation, m.groups, False).cuda()
        scale.weight.data.copy_(W_star)
        patch_size = m.kernel_size[0] * m.kernel_size[1]
        bias = 1.0/patch_size if m.bias is not None else 0
        return nn.Sequential(
            ConvLayerRotation(Q_a.t(), bias, trainable),
            scale,
            ConvLayerRotation(Q_g, trainable=trainable))
    else:
        raise NotImplementedError


# ====== for normalization ========
def normalize_factors(A, B):
    eps = 1e-10

    trA = torch.trace(A) + eps
    trB = torch.trace(B) + eps
    assert trA > 0, 'Must PD. A not PD'
    assert trB > 0, 'Must PD. B not PD'
    return A * (trB/trA)**0.5, B * (trA/trB)**0.5
