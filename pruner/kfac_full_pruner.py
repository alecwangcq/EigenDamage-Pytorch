import torch
import torch.nn as nn
from collections import OrderedDict
from models.resnet import _weights_init
from utils.kfac_utils import (ComputeCovA,
                              ComputeCovAPatch,
                              ComputeCovG,
                              fetch_mat_weights,
                              mat_to_weight_and_bias)
from utils.common_utils import (tensor_to_list, PresetLRScheduler)
from utils.prune_utils import (filter_indices,
                               get_threshold,
                               update_indices,
                               normalize_factors)
from utils.network_utils import stablize_bn
from tqdm import tqdm


class KFACFullPruner:

        def __init__(self,
                     model,
                     builder,
                     config,
                     writer,
                     logger,
                     prune_ratio_limit,
                     network,
                     batch_averaged=True,
                     use_patch=False,
                     fix_layers=0):
            print('Using patch is %s' % use_patch)
            self.iter = 0
            self.logger = logger
            self.writer = writer
            self.config = config
            self.prune_ratio_limit = prune_ratio_limit
            self.network = network
            self.CovAHandler = ComputeCovA() if not use_patch else ComputeCovAPatch()
            self.CovGHandler = ComputeCovG()
            self.batch_averaged = batch_averaged
            self.known_modules = {'Linear', 'Conv2d'}
            self.modules = []
            self.model = model
            self.builder = builder
            self.fix_layers = fix_layers
            # self._prepare_model()
            self.steps = 0
            self.use_patch = False  # use_patch
            self.m_aa, self.m_gg = {}, {}
            self.Q_a, self.Q_g = {}, {}
            self.d_a, self.d_g = {}, {}
            self.W_pruned = {}
            self.S_l = None

            self.importances = {}
            self._inversed = False
            self._cfgs = {}
            self._indices = {}

        def _save_input(self, module, input):
            aa = self.CovAHandler(input[0].data, module)
            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(0))
            self.m_aa[module] += aa

        def _save_grad_output(self, module, grad_input, grad_output):
            # Accumulate statistics for Fisher matrices
            gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(0))
            self.m_gg[module] += gg

        def make_pruned_model(self, dataloader, criterion, device, fisher_type, prune_ratio, normalize=True, re_init=False):
            self._prepare_model()
            self.init_step()

            self._compute_fisher(dataloader, criterion, device, fisher_type)
            self._update_inv()  # eigen decomposition of fisher

            self._get_unit_importance(normalize)
            self._do_prune(prune_ratio, re_init)
            if not re_init:
                self._do_surgery()
            self._build_pruned_model(re_init)

            self._rm_hooks()
            self._clear_buffer()
            print(self.model)
            return str(self.model)

        def _prepare_model(self):
            count = 0
            print(self.model)
            print("=> We keep following layers in KFACPruner. ")
            for module in self.model.modules():
                classname = module.__class__.__name__
                if classname in self.known_modules:
                    self.modules.append(module)
                    module.register_forward_pre_hook(self._save_input)
                    module.register_backward_hook(self._save_grad_output)
                    print('(%s): %s' % (count, module))
                    count += 1
            self.modules = self.modules[self.fix_layers:]

        def _compute_fisher(self, dataloader, criterion, device='cuda', fisher_type='true'):
            self.mode = 'basis'
            self.model = self.model.eval()
            self.init_step()
            for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                if fisher_type == 'true':
                    sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.cpu().data, dim=1),
                                                  1).squeeze().to(device)
                    loss_sample = criterion(outputs, sampled_y)
                    loss_sample.backward()
                else:
                    loss = criterion(outputs, targets)
                    loss.backward()
                self.step()
            self.mode = 'quite'

        def _update_inv(self):
            assert self.steps > 0, 'At least one step before update inverse!'
            eps = 1e-15
            for idx, m in enumerate(self.modules):
                # m_aa, m_gg = normalize_factors(self.m_aa[m], self.m_gg[m])
                m_aa, m_gg = self.m_aa[m], self.m_gg[m]
                self.d_a[m], self.Q_a[m] = torch.symeig(m_aa / self.steps, eigenvectors=True)
                self.d_g[m], self.Q_g[m] = torch.symeig(m_gg / self.steps, eigenvectors=True)
                self.d_a[m].mul_((self.d_a[m] > eps).float())
                self.d_g[m].mul_((self.d_g[m] > eps).float())

            self._inversed = True
            self.iter += 1

        def _get_unit_importance(self, normalize):
            eps = 1e-10
            assert self._inversed, 'Not inversed.'
            with torch.no_grad():
                for m in self.modules:
                    w = fetch_mat_weights(m, False)  # output_dim * input_dim
                    # (Q_a ⊗ Q_g) vec(W) = Q_g.t() @ W @ Q_a
                    if self.S_l is None:
                        A_inv = self.Q_a[m] @ (torch.diag(1.0 / (self.d_a[m] + eps))) @ self.Q_a[m].t()
                        G_inv = self.Q_g[m] @ (torch.diag(1.0 / (self.d_g[m] + eps))) @ self.Q_g[m].t()
                        A_inv_diag = torch.diag(A_inv)
                        G_inv_diag = torch.diag(G_inv)
                        w_imp = w ** 2 / (G_inv_diag.unsqueeze(1) @ A_inv_diag.unsqueeze(0))
                    else:
                        Q_a, Q_g = self.Q_a[m], self.Q_g[m]
                        S_l = self.S_l[m]
                        S_l_inv = 1.0 / (S_l + eps)
                        H_inv_diag = (Q_g ** 2) @ S_l_inv @ (Q_a.t() ** 2)  # output_dim * input_dim
                        w_imp = w ** 2 / H_inv_diag
                    self.W_pruned[m] = w
                    out_neuron_imp = w_imp.sum(1)  # w_imp.sum(1)
                    if not normalize:
                        out_imps = out_neuron_imp
                    else:
                        out_imps = out_neuron_imp / out_neuron_imp.sum()
                    self.importances[m] = (tensor_to_list(out_imps), out_neuron_imp.size(0))

        def _do_surgery(self):
            eps = 1e-10
            assert not self.use_patch, 'Will never use patch'
            with torch.no_grad():
                for idx, m in enumerate(self.modules):
                    w = fetch_mat_weights(m, False)  # output_dim * input_dim
                    if w.size(0) == len(m.out_indices):
                        continue
                    if self.S_l is None:
                        A_inv = self.Q_a[m] @ (torch.diag(1.0 / (self.d_a[m] + eps))) @ self.Q_a[m].t()
                        G_inv = self.Q_g[m] @ (torch.diag(1.0 / (self.d_g[m] + eps))) @ self.Q_g[m].t()
                        A_inv_diag = torch.diag(A_inv)
                        G_inv_diag = torch.diag(G_inv)
                        coeff = w / (G_inv_diag.unsqueeze(1) @ A_inv_diag.unsqueeze(0))
                        coeff[m.out_indices, :] = 0
                        delta_theta = -G_inv @ coeff @ A_inv
                    else:
                        Q_a, Q_g = self.Q_a[m], self.Q_g[m]
                        S_l = self.S_l[m]
                        S_l_inv = 1.0 / (S_l + eps)
                        H_inv_diag = (Q_g ** 2) @ S_l_inv @ (Q_a.t() ** 2)  # output_dim * input_dim
                        coeff = w / H_inv_diag
                        coeff[m.out_indices, :] = 0
                        delta_theta = (Q_g.t() @ coeff @ Q_a)/S_l_inv
                        delta_theta = Q_g @ delta_theta @ Q_a.t()
                    # ==== update weights and bias ======
                    dw, dbias = mat_to_weight_and_bias(delta_theta, m)
                    m.weight += dw
                    if m.bias is not None:
                        m.bias += dbias

        def _do_prune(self, prune_ratio, re_init):
            # get threshold
            all_importances = []
            for m in self.modules:
                imp_m = self.importances[m]
                imps = imp_m[0]
                all_importances += imps
            all_importances = sorted(all_importances)
            idx = int(prune_ratio * len(all_importances))
            threshold = all_importances[idx]

            threshold_recompute = get_threshold(all_importances, prune_ratio)
            idx_recomputed = len(filter_indices(all_importances, threshold))
            print('=> The threshold is: %.5f (%d), computed by function is: %.5f (%d).' % (threshold,
                                                                                           idx,
                                                                                           threshold_recompute,
                                                                                           idx_recomputed))

            # do pruning
            print('=> Conducting network pruning. Max: %.5f, Min: %.5f, Threshold: %.5f' % (max(all_importances),
                                                                                            min(all_importances),
                                                                                            threshold))
            self.logger.info("[Weight Importances] Max: %.5f, Min: %.5f, Threshold: %.5f." % (max(all_importances),
                                                                                              min(all_importances),
                                                                                              threshold))

            for idx, m in enumerate(self.modules):
                imp_m = self.importances[m]
                n_r = imp_m[1]
                row_imps = imp_m[0]
                row_indices = filter_indices(row_imps, threshold)
                r_ratio = 1 - len(row_indices) / n_r

                # compute row indices (out neurons)
                if r_ratio > self.prune_ratio_limit:
                    r_threshold = get_threshold(row_imps, self.prune_ratio_limit)
                    row_indices = filter_indices(row_imps, r_threshold)  # list(range(self.W_star[m].size(0)))
                    print('* row indices empty!')
                if isinstance(m, nn.Linear) and idx == len(self.modules) - 1:
                    row_indices = list(range(self.W_pruned[m].size(0)))

                m.out_indices = row_indices
                m.in_indices = None
            update_indices(self.model, self.network)

        def _build_pruned_model(self, re_init):
            for m in self.model.modules():
                # m.grad = None
                if isinstance(m, nn.BatchNorm2d):
                    idxs = m.in_indices
                    m.num_features = len(idxs)
                    m.weight.data = m.weight.data[idxs]
                    m.bias.data = m.bias.data[idxs].clone()
                    m.running_mean = m.running_mean[idxs].clone()
                    m.running_var = m.running_var[idxs].clone()
                    # m.in_indices = None
                    # m.out_indices = None
                    m.weight.grad = None
                    m.bias.grad = None
                elif isinstance(m, nn.Conv2d):
                    in_indices = m.in_indices
                    if m.in_indices is None: 
                        in_indices = list(range(m.weight.size(1)))
                    m.weight.data = m.weight.data[m.out_indices, :, :, :][:, in_indices, :, :].clone()
                    if m.bias is not None:
                        m.bias.data = m.bias.data[m.out_indices]
                        m.bias.grad = None
                    m.in_channels = len(in_indices)
                    m.out_channels = len(m.out_indices)
                    # m.in_indices = None
                    # m.out_indices = None
                    m.weight.grad = None
                    
                elif isinstance(m, nn.Linear):
                    in_indices = m.in_indices
                    if m.in_indices is None:
                        in_indices = list(range(m.weight.size(1)))
                    m.weight.data = m.weight.data[m.out_indices, :][:, in_indices].clone()
                    if m.bias is not None:
                        m.bias.data = m.bias.data[m.out_indices].clone()
                        m.bias.grad = None
                    m.in_features = len(in_indices)
                    m.out_features = len(m.out_indices)
                    # m.in_indices = None
                    # m.out_indices = None
                    m.weight.grad = None
            if re_init:
                self.model.apply(_weights_init)
            # import pdb; pdb.set_trace()

        def init_step(self):
            self.steps = 0

        def step(self):
            self.steps += 1

        def _rm_hooks(self):
            for m in self.model.modules():
                classname = m.__class__.__name__
                if classname in self.known_modules:
                    m._backward_hooks = OrderedDict()
                    m._forward_pre_hooks = OrderedDict()

        def _clear_buffer(self):
            self.m_aa = {}
            self.m_gg = {}
            self.d_a = {}
            self.d_g = {}
            self.Q_a = {}
            self.Q_g = {}
            self.modules = []
            if self.S_l is not None:
                self.S_l = {}

        def fine_tune_model(self, trainloader, testloader, criterion, optim, learning_rate, weight_decay, nepochs=10,
                            device='cuda'):
            self.model = self.model.train()
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
            # optimizer = optim.Adam(self.model.parameters(), weight_decay=5e-4)
            lr_schedule = {0: learning_rate, int(nepochs * 0.5): learning_rate * 0.1,
                           int(nepochs * 0.75): learning_rate * 0.01}
            lr_scheduler = PresetLRScheduler(lr_schedule)
            best_test_acc, best_test_loss = 0, 100
            iterations = 0
            for epoch in range(nepochs):
                self.model = self.model.train()
                correct = 0
                total = 0
                all_loss = 0
                lr_scheduler(optimizer, epoch)
                desc = ('[LR: %.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                lr_scheduler.get_lr(optimizer), 0, 0, correct, total))
                prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
                for batch_idx, (inputs, targets) in prog_bar:
                    optimizer.zero_grad()
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    self.writer.add_scalar('train_%d/loss' % self.iter, loss.item(), iterations)
                    iterations += 1
                    all_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    desc = ('[%d][LR: %.5f, WD: %.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                            (epoch, lr_scheduler.get_lr(optimizer), weight_decay, all_loss / (batch_idx + 1),
                             100. * correct / total, correct, total))
                    prog_bar.set_description(desc, refresh=True)
                test_loss, test_acc = self.test_model(testloader, criterion, device)
                best_test_loss = best_test_loss if best_test_acc > test_acc else test_loss
                best_test_acc = max(test_acc, best_test_acc)
            print('** Finetuning finished. Stabilizing batch norm and test again!')
            stablize_bn(self.model, trainloader)
            test_loss, test_acc = self.test_model(testloader, criterion, device)
            best_test_loss = best_test_loss if best_test_acc > test_acc else test_loss
            best_test_acc = max(test_acc, best_test_acc)
            return best_test_loss, best_test_acc

        def test_model(self, dataloader, criterion, device='cuda'):
            self.model = self.model.eval()
            correct = 0
            total = 0
            all_loss = 0
            desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (0, 0, correct, total))
            prog_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=desc, leave=True)
            for batch_idx, (inputs, targets) in prog_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                all_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                        (all_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                prog_bar.set_description(desc, refresh=True)
            return all_loss / (batch_idx + 1), 100. * correct / total
