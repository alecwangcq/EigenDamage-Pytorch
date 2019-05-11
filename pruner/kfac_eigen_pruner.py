import torch
import torch.nn as nn
from collections import OrderedDict
from utils.kfac_utils import (ComputeCovA,
                              ComputeCovG,
                              ComputeCovAPatch,
                              fetch_mat_weights)
from utils.common_utils import (tensor_to_list,
                                PresetLRScheduler)
from utils.prune_utils import (count_module_params,
                               get_rotation_layer_weights,
                               get_threshold,
                               filter_indices,
                               normalize_factors)
from utils.network_utils import stablize_bn
from tqdm import tqdm


class KFACEigenPruner:

    def __init__(self,
                 model,
                 builder,
                 config,
                 writer,
                 logger,
                 prune_ratio_limit,
                 batch_averaged=True,
                 fix_rotation=True,
                 use_patch=False,
                 fix_layers=0):
        print('Using patch is %s' % use_patch)
        self.iter = 0
        self.logger = logger
        self.writer = writer
        self.config = config
        self.prune_ratio_limit = prune_ratio_limit
        self.CovAHandler = ComputeCovA() if not use_patch else ComputeCovAPatch()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged
        self.known_modules = {'Linear', 'Conv2d'}
        self.modules = []
        self.grad_outputs = {}
        self.model = model
        self.builder = builder
        self.fix_layers = fix_layers
        # self._prepare_model()
        self.steps = 0
        self.use_patch = use_patch
        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.W_star = {}
        self.S_l = None

        self.importances = {}
        self._inversed = False
        self.fix_rotation = fix_rotation

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

    def _merge_Qs(self):
        for m, v in self.Q_g.items():
            if len(v) > 1:
                self.Q_g[m] = v[1] @ v[0]
            else:
                self.Q_g[m] = v[0]
        for m, v in self.Q_a.items():
            if len(v) > 1:
                self.Q_a[m] = v[1] @ v[0]
            else:
                self.Q_a[m] = v[0]

    def make_pruned_model(self, dataloader, criterion, device, fisher_type, prune_ratio, normalize=True, re_init=False):
        self._prepare_model()
        self.init_step()

        self._compute_fisher(dataloader, criterion, device, fisher_type)
        self._update_inv()  # eigen decomposition of fisher

        self._get_unit_importance(normalize)
        self._merge_Qs()  # update the eigen basis

        self._do_prune(prune_ratio)

        self._rm_hooks()
        self._build_pruned_model(re_init)
        self._clear_buffer()

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
        self.modules = self.modules[self.fix_layers:-1]

    def _compute_fisher(self, dataloader, criterion, device='cuda', fisher_type='true'):
        self.model = self.model.eval()
        self.init_step()
        for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = self.model(inputs)
            if fisher_type == 'true':
                sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.cpu().data, dim=1), 1).squeeze().to(device)
                loss_sample = criterion(outputs, sampled_y)
                loss_sample.backward()
            else:
                loss = criterion(outputs, targets)
                loss.backward()
            self.step()

    def _update_inv(self):
        assert self.steps > 0, 'At least one step before update inverse!'
        eps = 1e-10
        for idx, m in enumerate(self.modules):
            m_aa, m_gg = self.m_aa[m], self.m_gg[m]
            self.d_a[m], Q_a = torch.symeig(m_aa, eigenvectors=True)
            self.d_g[m], Q_g = torch.symeig(m_gg, eigenvectors=True)

            self.d_a[m].mul_((self.d_a[m] > eps).float())
            self.d_g[m].mul_((self.d_g[m] > eps).float())

            # == write summary ==
            name = m.__class__.__name__
            eigs = (self.d_g[m].view(-1, 1) @ self.d_a[m].view(1, -1)).view(-1).cpu().data.numpy()
            self.writer.add_histogram('eigen/%s_%d' % (name, idx), eigs, self.iter)

            if self.Q_a.get(m, None) is None:
                # print('(%d)Q_a %s is None.' % (idx, m))
                self.Q_a[m] = [Q_a]  # absorb the eigen basis
            else:
                # self.Q_a[m] = [Q_a, self.Q_a[m]]
                prev_Q_a, prev_Q_g = get_rotation_layer_weights(self.model, m)
                prev_Q_a = prev_Q_a.view(prev_Q_a.size(0), prev_Q_a.size(1)).transpose(1, 0)
                prev_Q_g = prev_Q_g.view(prev_Q_g.size(0), prev_Q_g.size(1))
                self.Q_a[m] = [Q_a, prev_Q_a]

            if self.Q_g.get(m, None) is None:
                self.Q_g[m] = [Q_g]
            else:
                self.Q_g[m] = [Q_g, prev_Q_g]
        self._inversed = True
        self.iter += 1

    def _get_unit_importance(self, normalize):
        assert self._inversed, 'Not inversed.'
        with torch.no_grad():
            for m in self.modules:
                w = fetch_mat_weights(m, self.use_patch)  # output_dim * input_dim
                # (Q_a ⊗ Q_g) vec(W) = Q_g.t() @ W @ Q_a
                if self.use_patch and isinstance(m, nn.Conv2d):
                    w_star_a = w.view(-1, w.size(-1)) @ self.Q_a[m][0]
                    w_star_g = self.Q_g[m][0].t() @ w_star_a.view(w.size(0), -1)
                    w_star = w_star_g.view(w.size())
                    if self.S_l is None:
                        w_imp = w_star ** 2 * (self.d_g[m].unsqueeze(1) @ self.d_a[m].unsqueeze(0)).unsqueeze(1)
                    else:
                        w_imp = w_star ** 2 * self.S_l[m]
                    w_imp = w_imp.sum(dim=1)
                else:
                    w_star = self.Q_g[m][0].t() @ w @ self.Q_a[m][0]
                    if self.S_l is None:
                        w_imp = w_star ** 2 * (self.d_g[m].unsqueeze(1) @ self.d_a[m].unsqueeze(0))
                    else:
                        w_imp = w_star ** 2 * self.S_l[m]

                self.W_star[m] = w_star
                in_neuron_imp = w_imp.sum(0)  # get_block_sum(m, w_imp.sum(0))
                out_neuron_imp = w_imp.sum(1)  # w_imp.sum(1)
                if not normalize:
                    imps = torch.cat([in_neuron_imp, out_neuron_imp])
                else:
                    # I found in most cases normalization will harm the performance.
                    imps = torch.cat([in_neuron_imp/in_neuron_imp.sum(), out_neuron_imp/out_neuron_imp.sum()])
                self.importances[m] = (tensor_to_list(imps), in_neuron_imp.size(0),
                                       out_neuron_imp.size(0), len(self.Q_g[m]) == 2)

    def _do_prune(self, prune_ratio):
        # get threshold
        all_importances = []
        for m in self.modules:
            imp_m = self.importances[m]
            imps = imp_m[0]
            all_importances += imps
        all_importances = sorted(all_importances)
        idx = int(prune_ratio * len(all_importances))
        threshold = all_importances[idx]

        # do pruning
        print('=> Conducting network pruning. Max: %.5f, Min: %.5f, Threshold: %.5f' % (max(all_importances),
                                                                                        min(all_importances),
                                                                                        threshold))
        self.logger.info("[Weight Improtances] Max: %.5f, Min: %.5f, Threshold: %.5f." % (max(all_importances),
                                                                                          min(all_importances),
                                                                                          threshold))
        total_remain = 0
        total_origin = 0
        fake_remain = 0
        self.remain_modules = []
        for m in self.modules:
            imp_m = self.importances[m]
            imps, n_c, n_r, is_pruned = imp_m[0], imp_m[1], imp_m[2], imp_m[3]
            row_indices = []
            col_indices = []
            for i in range(n_r):
                if imps[n_c + i] > threshold:
                    row_indices.append(i)
            for i in range(n_c):
                if imps[i] > threshold:
                    col_indices.append(i)

            r_ratio = 1 - len(row_indices) / n_r
            c_ratio = 1 - len(col_indices) / n_c
            if r_ratio > self.prune_ratio_limit:
                r_threshold = get_threshold(imps[n_c:], self.prune_ratio_limit)
                row_indices = filter_indices(imps[n_c:], r_threshold)
                print('* row indices empty!')
            if c_ratio > self.prune_ratio_limit:
                c_threshold = get_threshold(imps[:n_c], self.prune_ratio_limit)
                col_indices = filter_indices(imps[:n_c], c_threshold)
                print('* col indices empty!')
            # assert len(row_indices) > 0 and len(col_indices) > 0, "Resulted in empty tensor!"
            # ===========================================================================
            # If it is pruned, then we take the size of bottleneck as the layer size.
            # ===========================================================================
            origin = count_module_params(m) if not is_pruned else (self.Q_a[m].numel() + self.Q_g[m].numel() +
                                                                   self.W_star[m].numel())

            # ===========================================================================
            # start pruning
            # ===========================================================================
            self.Q_a[m] = self.Q_a[m][:, col_indices]
            self.Q_g[m] = self.Q_g[m][:, row_indices]
            # try:
            self.W_star[m] = self.W_star[m][row_indices, :][..., col_indices]

            # ===========================================================================
            # get the number of parameters in the pruned bottleneck layer.
            # ===========================================================================
            bneck_counts = (self.Q_a[m].numel() + self.Q_g[m].numel() + self.W_star[m].numel())
            total_origin += origin
            exact_ratio = bneck_counts / origin

            # ===========================================================================
            # if the bottleneck size is larger than the
            # original one, we do not conduct pruning.
            # ===========================================================================
            if exact_ratio <= 1 or (exact_ratio == 1 and is_pruned):
                marker = 'o'
                total_remain += bneck_counts
            else:
                marker = 'x'
                total_remain += origin

            # ============================================================================
            # fake_reamin is the size of W matrix. Without considering the rotation matrix
            # ============================================================================
            fake_remain += self.W_star[m].numel()
            all_ratio = (len(col_indices) * len(row_indices)) / (self.Q_a[m].size(0) * self.Q_g[m].size(0))
            row_ratio = len(row_indices) / (self.Q_g[m].size(0))
            col_ratio = len(col_indices) / (self.Q_a[m].size(0))
            if exact_ratio > 1:
                self.Q_g.pop(m)
                self.Q_a.pop(m)
            print('[%s]Pruning: %-3.2f%%(w/o Q), %-3.2f%% | %-3.2f%%(exact), '
                  '%-3.2f%% (w/o Q), %-3.2f%%(out_neuron), %-3.2f%%(in_neuron): %s' %
                  (marker, 100*fake_remain/total_origin, 100*total_remain/total_origin,
                   100*exact_ratio, 100*all_ratio, 100*row_ratio, 100*col_ratio, m))
            if exact_ratio <= 1 or (exact_ratio == 1 and is_pruned):
                self.remain_modules.append(m)

    def _build_pruned_model(self, re_init):
        self.model = self.builder(self.model, self.fix_rotation)
        self.model.register(self.remain_modules,
                            self.Q_g, self.Q_a,
                            self.W_star,
                            self.use_patch,
                            fix_rotation=not self.fix_rotation, re_init=re_init)

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
        self.remain_modules = []
        self.modules = []
        self.W_star = {}
        if self.S_l is not None:
            self.S_l = {}

    def fine_tune_model(self, trainloader, testloader, criterion, optim, learning_rate, weight_decay, nepochs=10, device='cuda'):
        self.model = self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        lr_schedule = {0: learning_rate, int(nepochs*0.5): learning_rate*0.1, int(nepochs*0.75): learning_rate*0.01}
        lr_scheduler = PresetLRScheduler(lr_schedule)
        best_test_acc, best_test_loss = 0, 100
        iterations = 0
        for epoch in range(nepochs):
            self.model = self.model.train()
            correct = 0
            total = 0
            all_loss = 0
            lr_scheduler(optimizer, epoch)
            desc = ('[LR: %.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' % (lr_scheduler.get_lr(optimizer), 0, 0, correct, total))
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
            #try:
            outputs = self.model(inputs)
            #except:
            #    import pdb; pdb.set_trace()
            loss = criterion(outputs, targets)
            all_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                    (all_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)
        return all_loss / (batch_idx + 1), 100. * correct / total
