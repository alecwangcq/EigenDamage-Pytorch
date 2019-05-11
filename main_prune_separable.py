import argparse
import json
import os
import sys
import torch
import torch.optim as optim

from models import VGG
from pruner.fisher_diag_pruner import FisherDiagPruner
from pruner.kfac_eigen_pruner import KFACEigenPruner
from pruner.kfac_full_pruner import KFACFullPruner
from pruner.kfac_OBD_F2 import KFACOBDF2Pruner
from pruner.kfac_OBS_F2 import KFACOBSF2Pruner
from pruner.kfac_eigen_svd_pruner import KFACEigenSVDPruner

from tensorboardX import SummaryWriter
from utils.common_utils import (get_config_from_json,
                                get_logger,
                                makedirs,
                                process_config,
                                str_to_list)
from utils.data_utils import get_dataloader
from utils.network_utils import (get_bottleneck_builder,
                                 get_network)
from utils.prune_utils import (ConvLayerRotation,
                               LinearLayerRotation)

from utils.compute_flops import print_model_param_flops


def count_parameters(model):
    """The number of trainable parameters.
    It will exclude the rotation matrix in bottleneck layer.
    If those parameters are not trainiable.
    """
    return sum(p.numel() for p in model.parameters())


def count_rotation_numels(model):
    """Count how many parameters in the rotation matrix.
    Call this only when they are not trainable for complementing
    the number of parameters.
    """
    total = 0
    for m in model.modules():
        if isinstance(m, (ConvLayerRotation, LinearLayerRotation)):
            total += m.rotation_matrix.numel()
    return total


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = process_config(args.config)

    return config


def init_dataloader(config):
    trainloader, testloader = get_dataloader(dataset=config.dataset,
                                             train_batch_size=config.batch_size,
                                             test_batch_size=128)
    return trainloader, testloader


def init_network(config, logger, device):
    net = get_network(network=config.network,
                      depth=config.depth,
                      dataset=config.dataset)
    print('==> Loading checkpoint from %s.' % config.checkpoint)
    logger.info('==> Loading checkpoint from %s.' % config.checkpoint)
    checkpoint = torch.load(config.checkpoint)
    if checkpoint.get('args', None) is not None:
        args = checkpoint['args']
        print('** [%s-%s%d] Acc: %.2f%%, Epoch: %d, Loss: %.4f' % (args.dataset, args.network, args.depth,
                                                                   checkpoint['acc'], checkpoint['epoch'],
                                                                   checkpoint['loss']))
        logger.info('** [%s-%s%d] Acc: %.2f%%, Epoch: %d, Loss: %.4f' % (args.dataset, args.network, args.depth,
                                                                         checkpoint['acc'], checkpoint['epoch'],
                                                                         checkpoint['loss']))
    state_dict = checkpoint['net'] if checkpoint.get('net', None) is not None else checkpoint['state_dict']
    net.load_state_dict(state_dict)
    bottleneck_net = get_bottleneck_builder(config.network)

    return net.to(device), bottleneck_net


def init_pruner(net, bottleneck_net, config, writer, logger):
    if config.fisher_mode == 'eigen':
        pruner = KFACEigenPruner(net,
                                 bottleneck_net,
                                 config,
                                 writer,
                                 logger,
                                 config.prune_ratio_limit,
                                 batch_averaged=True,
                                 use_patch=config.get('use_patch', True),
                                 fix_layers=config.fix_layers,
                                 fix_rotation=config.fix_rotation)
    elif config.fisher_mode == 'full':
        pruner = KFACFullPruner(net,
                                VGG,
                                config,
                                writer,
                                logger,
                                config.prune_ratio_limit,
                                '%s%d'%(config.network, config.depth),
                                batch_averaged=True,
                                use_patch=False,
                                fix_layers=0)
    elif config.fisher_mode == 'diag':
        # FisherDiagPruner
        pruner = FisherDiagPruner(model=net,
                                  builder=VGG,
                                  config=config,
                                  writer=writer,
                                  logger=logger,
                                  prune_ratio_limit=config.prune_ratio_limit,
                                  network='%s%d' % (config.network, config.depth),
                                  batch_averaged=True,
                                  use_patch=False,
                                  fix_layers=0)
    elif config.fisher_mode == 'OBD_F2':
        pruner = KFACOBDF2Pruner(net,
                                 VGG,
                                 config,
                                 writer,
                                 logger,
                                 config.prune_ratio_limit,
                                 '%s%d'%(config.network, config.depth),
                                 batch_averaged=True,
                                 use_patch=False,
                                 fix_layers=0)
    elif config.fisher_mode == 'OBS_F2':
        pruner = KFACOBSF2Pruner(net,
                                 VGG,
                                 config,
                                 writer,
                                 logger,
                                 config.prune_ratio_limit,
                                 '%s%d'%(config.network, config.depth),
                                 batch_averaged=True,
                                 use_patch=False,
                                 fix_layers=0)
    elif config.fisher_mode == 'eigen_svd':
        pruner = KFACEigenSVDPruner(net,
                                    bottleneck_net,
                                    config,
                                    writer,
                                    logger,
                                    config.prune_ratio_limit,
                                    batch_averaged=True,
                                    use_patch=config.get('use_patch', True),
                                    fix_layers=config.fix_layers,
                                    fix_rotation=config.fix_rotation)
    else:
        raise NotImplementedError

    return pruner


def init_summary_writer(config):
    makedirs(config.summary_dir)
    makedirs(config.checkpoint_dir)

    # set logger
    path = os.path.dirname(os.path.abspath(__file__))
    path_model = os.path.join(path, 'models/%s.py' % config.network)
    path_main = os.path.join(path, 'main_prune.py')
    path_pruner = os.path.join(path, 'pruner/%s.py' % config.pruner)
    logger = get_logger('log', logpath=config.summary_dir+'/',
                        filepath=path_model, package_files=[path_main, path_pruner])
    logger.info(dict(config))
    writer = SummaryWriter(config.summary_dir)

    return logger, writer


def save_model(config, iteration, pruner, cfg, stat, fintuned=True):
    network = config.network
    depth = config.depth
    dataset = config.dataset
    checkpoint_dir = config.checkpoint_dir
    if fintuned:
        path = os.path.join(checkpoint_dir, '%s_%s%s_%d.pth.tar' % (dataset, network, depth, iteration))
    else:
        path = os.path.join(checkpoint_dir, 'pre_%s_%s%s_%d.pth.tar' % (dataset, network, depth, iteration))
    save = {
        # 'Q_a': pruner.Q_a,
        # 'Q_g': pruner.Q_g,
        'config': config,
        'net': pruner.model,
        'cfg': cfg,
        'stat': stat
    }
    torch.save(save, path)


def load_model(config, iteration, fintuned):
    network = config.network
    depth = config.depth
    dataset = config.dataset
    checkpoint_dir = config.checkpoint_dir
    if fintuned:
        path = os.path.join(checkpoint_dir, '%s_%s%s_%d.pth.tar' % (dataset, network, depth, iteration))
    else:
        path = os.path.join(checkpoint_dir, 'pre_%s_%s%s_%d.pth.tar' % (dataset, network, depth, iteration))
    print('=> Loading from %s.' % path)
    state = torch.load(path)
    print(state['net'])
    print('Loaded.')
    return state


def compute_ratio(model, total, fix_rotation, logger):
    indicator = 1 if fix_rotation else 0
    rotation_numel = count_rotation_numels(model)
    pruned_numel = count_parameters(model) + rotation_numel*indicator
    ratio = 100. * pruned_numel / total
    logger.info('Compression ratio: %.2f%%(%d/%d), Total: %d, Rotation: %d.' % (ratio,
                                                                                pruned_numel,
                                                                                total,
                                                                                pruned_numel,
                                                                                rotation_numel))
    unfair_ratio = 100 - 100. * (pruned_numel - rotation_numel*indicator)
    return 100 - ratio, unfair_ratio, pruned_numel, rotation_numel


def main(config):
    stats = {}
    device = 'cuda'
    criterion = torch.nn.CrossEntropyLoss()

    # config = init_config() if config is None else config
    logger, writer = init_summary_writer(config)
    trainloader, testloader = init_dataloader(config)
    net, bottleneck_net = init_network(config, logger, device)
    pruner = init_pruner(net, bottleneck_net, config, writer, logger)

    # start pruning
    epochs = str_to_list(config.epoch, ',', int)
    learning_rates = str_to_list(config.learning_rate, ',', float)
    weight_decays = str_to_list(config.weight_decay, ',', float)
    ratios = str_to_list(config.ratio, ',', float)
    if config.get('pre_model', None) is not None:
        pre_load = str_to_list(config.pre_model, ',', int)
    else:
        pre_load = None

    fisher_type = config.fisher_type  # empirical|true
    fisher_mode = config.fisher_mode  # eigen|full|diagonal
    normalize = config.normalize
    prune_mode = config.prune_mode  # one-pass | iterative
    fix_rotation = config.get('fix_rotation', True)

    assert (len(epochs) == len(learning_rates) and
            len(learning_rates) == len(weight_decays) and
            len(weight_decays) == len(ratios))

    total_parameters = count_parameters(net.train())
    for it in range(len(epochs)):
        epoch = epochs[it]
        lr = learning_rates[it]
        wd = weight_decays[it]
        ratio = ratios[it]
        logger.info('-'*120)
        logger.info('** [%d], Ratio: %.2f, epoch: %d, lr: %.4f, wd: %.4f' % (it, ratio, epoch, lr, wd))
        logger.info('Reinit: %s, Fisher_mode: %s, fisher_type: %s, normalize: %s, fix_rotation: %s.' % (config.re_init,
                                                                                                        fisher_mode,
                                                                                                        fisher_type,
                                                                                                        normalize,
                                                                                                        fix_rotation))
        pruner.fix_rotation = fix_rotation

        # conduct pruning
        if pre_load is not None and pre_load[it] == 1:
            state = load_model(config, it, False)
            pruner.model = state['net']
            cfg = state['cfg']
        else:
            cfg = pruner.make_pruned_model(trainloader,
                                           criterion=criterion,
                                           device=device,
                                           fisher_type=fisher_type,
                                           prune_ratio=ratio,
                                           normalize=normalize,
                                           re_init=config.re_init)
            save_model(config, it, pruner, cfg, None, False)

        # for tracking the best accuracy
        compression_ratio, unfair_ratio, all_numel, rotation_numel = compute_ratio(pruner.model, total_parameters,
                                                                                   fix_rotation, logger)

        if config.dataset == 'tiny_imagenet':
            total_flops, rotation_flops = print_model_param_flops(pruner.model, 64, cuda=True)
        else:
            total_flops, rotation_flops = print_model_param_flops(pruner.model, 32, cuda=True)
        train_loss_pruned, train_acc_pruned = pruner.test_model(trainloader, criterion, device)
        test_loss_pruned, test_acc_pruned = pruner.test_model(testloader, criterion, device)

        # write results
        logger.info('Before: Accuracy: %.2f%%(train), %.2f%%(test).' % (train_acc_pruned, test_acc_pruned))
        logger.info('        Loss:     %.2f  (train), %.2f  (test).' % (train_loss_pruned, test_loss_pruned))

        test_loss_finetuned, test_acc_finetuned = pruner.fine_tune_model(trainloader=trainloader,
                                                                         testloader=testloader,
                                                                         criterion=criterion,
                                                                         optim=optim,
                                                                         learning_rate=lr,
                                                                         weight_decay=wd,
                                                                         nepochs=epoch)
        train_loss_finetuned, train_acc_finetuned = pruner.test_model(trainloader, criterion, device)
        logger.info('After:  Accuracy: %.2f%%(train), %.2f%%(test).' % (train_acc_finetuned, test_acc_finetuned))
        logger.info('        Loss:     %.2f  (train), %.2f  (test).' % (train_loss_finetuned, test_loss_finetuned))
        # save model

        stat = {
            'total_flops': total_flops,
            'rotation_flops': rotation_flops,
            'it': it,
            'prune_ratio': ratio,
            'cr': compression_ratio,
            'unfair_cr': unfair_ratio,
            'all_params': all_numel,
            'rotation_params': rotation_numel,
            'prune/train_loss': train_loss_pruned,
            'prune/train_acc': train_acc_pruned,
            'prune/test_loss': test_loss_pruned,
            'prune/test_acc': test_acc_pruned,
            'finetune/train_loss': train_loss_finetuned,
            'finetune/test_loss': test_loss_finetuned,
            'finetune/train_acc': train_acc_finetuned,
            'finetune/test_acc': test_acc_finetuned
        }
        save_model(config, it, pruner, cfg, stat)

        stats[it] = stat

        if prune_mode == 'one_pass':
            del net
            del pruner
            net, bottleneck_net = init_network(config, logger, device)
            pruner = init_pruner(net, bottleneck_net, config, writer, logger)
            pruner.iter = it
        with open(os.path.join(config.summary_dir, 'stats.json'), 'w') as f:
            json.dump(stats, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmp_config', type=str, default='', required=False)
    parser.add_argument('--config', type=str, default='', required=False)
    args = parser.parse_args()

    if len(args.tmp_config) > 1:
        print('Using tmp config!')
        config, _ = get_config_from_json(args.tmp_config)
        makedirs(config.summary_dir)
        sys.stdout = open(os.path.join(config.summary_dir, 'stdout.txt'), 'w+')
        sys.stderr = open(os.path.join(config.summary_dir, 'stderr.txt'), 'w+')
        main(config)
    else:
        print('Using config!')
        config = process_config(args.config)
        main(config)