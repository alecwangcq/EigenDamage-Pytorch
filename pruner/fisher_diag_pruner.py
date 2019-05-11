import torch

from pruner.kfac_full_pruner import KFACFullPruner
from utils.common_utils import tensor_to_list
from utils.kfac_utils import (ComputeMatGrad,
                              fetch_mat_weights)


class FisherDiagPruner(KFACFullPruner):

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
        self.use_patch = False
        self.logger = logger
        self.writer = writer
        self.config = config
        self.prune_ratio_limit = prune_ratio_limit
        self.network = network
        self.MatGradHandler = ComputeMatGrad()

        self.batch_averaged = batch_averaged
        self.known_modules = {'Linear', 'Conv2d'}
        self.modules = []
        self.model = model
        self.builder = builder
        self.fix_layers = fix_layers
        self.steps = 0
        self.W_pruned = {}

        self.importances = {}
        self._inversed = False
        self._cfgs = {}
        self._indices = {}

        self.A, self.DS = {}, {}
        self.Fisher = {}

    def _save_input(self, module, input):
        self.A[module] = input[0].data

    def _save_grad_output(self, module, grad_input, grad_output):
        self.DS[module] = grad_output[0].data

    def make_pruned_model(self, dataloader, criterion, device, fisher_type, prune_ratio, normalize=True, re_init=False):
        self._prepare_model()
        self.init_step()

        temp_loader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=64, shuffle=True,
                                                  num_workers=2)
        self._compute_fisher(temp_loader, criterion, device, fisher_type)

        self.iter += 1
        self._get_unit_importance(normalize)
        self._do_prune(prune_ratio, re_init)
        self._build_pruned_model(re_init)

        self._rm_hooks()
        # cfg = self._build_pruned_model(re_init)
        self._clear_buffer()
        return str(self.model)

    def _get_unit_importance(self, normalize):
        eps = 1e-10
        with torch.no_grad():
            for m in self.modules:
                w = fetch_mat_weights(m, False)  # output_dim * input_dim
                F_diag = (self.Fisher[m] / self.steps + eps)
                w_imp = w ** 2 * F_diag
                self.W_pruned[m] = w
                out_neuron_imp = w_imp.sum(1)  # w_imp.sum(1)
                if not normalize:
                    imps = out_neuron_imp
                else:
                    imps = out_neuron_imp / out_neuron_imp.sum()
                self.importances[m] = (tensor_to_list(imps), out_neuron_imp.size(0))

    def step(self):
        with torch.no_grad():
            for m in self.modules:
                A, DS = self.A[m], self.DS[m]
                grad_mat = self.MatGradHandler(A, DS, m)
                if self.batch_averaged:
                    grad_mat *= DS.size(0)
                if self.steps == 0:
                    self.Fisher[m] = grad_mat.new(grad_mat.size()[1:]).fill_(0)
                self.Fisher[m] += (grad_mat.pow_(2)).sum(0)
                self.A[m] = None
                self.DS[m] = None
        self.steps += 1

    def _clear_buffer(self):
        self.Fisher = {}
        self.modules = []
