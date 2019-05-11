"""
F2 = A ⊗ B,
A = in_c * in_c
B = out_c * out_c  (Diagonal)
"""
import torch

from pruner.kfac_full_pruner import KFACFullPruner
from utils.common_utils import tensor_to_list
from utils.kfac_utils import (ComputeMatGrad,
                              fetch_mat_weights)


class KFACOBDF2Pruner(KFACFullPruner):
    def __init__(self, *args, **kwargs):
        super(KFACOBDF2Pruner, self).__init__(*args, **kwargs)
        print("Using OBD F2.")

    def _get_unit_importance(self, normalize):
        with torch.no_grad():
            for m in self.modules:
                w = fetch_mat_weights(m, False)  # output_dim * input_dim
                w_imp = w**2 @ self.m_aa[m]
                out_neuron_imp = w_imp.sum(1) * torch.diag(self.m_gg[m])
                self.W_pruned[m] = w
                if not normalize:
                    imps = out_neuron_imp
                else:
                    imps = out_neuron_imp / out_neuron_imp.sum()
                self.importances[m] = (tensor_to_list(imps), out_neuron_imp.size(0))

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