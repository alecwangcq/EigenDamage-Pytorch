"""
F2 = A ⊗ B,
A = in_c * in_c
B = out_c * out_c  (Diagonal)
"""
import torch

from pruner.kfac_full_pruner import KFACFullPruner
from utils.common_utils import tensor_to_list
from utils.kfac_utils import (fetch_mat_weights,
                              mat_to_weight_and_bias)


class KFACOBSF2Pruner(KFACFullPruner):
    def __init__(self, *args, **kwargs):
        super(KFACOBSF2Pruner, self).__init__(*args, **kwargs)
        print("Using OBS F2.")

    def _get_unit_importance(self, normalize):
        eps = 1e-10
        with torch.no_grad():
            for m in self.modules:
                w = fetch_mat_weights(m, False)  # output_dim * input_dim
                G_inv = self.Q_g[m] @ (torch.diag(1.0 / (self.d_g[m] + eps))) @ self.Q_g[m].t()
                w_imps = torch.sum(w**2@self.m_aa[m], 1)  # output_dim
                out_neuron_imp = w_imps / torch.diag(G_inv)
                self.W_pruned[m] = w
                if not normalize:
                    out_imps = out_neuron_imp
                else:
                    out_imps = out_neuron_imp / out_neuron_imp.sum()
                self.importances[m] = (tensor_to_list(out_imps), out_neuron_imp.size(0))

    def _do_surgery(self):
        eps = 1e-10
        with torch.no_grad():
            for idx, m in enumerate(self.modules):
                w = fetch_mat_weights(m, False)
                if w.size(0) == len(m.out_indices):
                    continue
                G_inv = self.Q_g[m] @ (torch.diag(1.0 / (self.d_g[m] + eps))) @ self.Q_g[m].t()
                G_inv_diag = torch.diag(G_inv)
                G_inv[:, m.out_indices] = 0
                coeff = G_inv @ torch.diag(1.0 / G_inv_diag)
                delta_theta = -coeff @ w

                # ==== update weights and bias ======
                dw, dbias = mat_to_weight_and_bias(delta_theta, m)
                m.weight += dw
                if m.bias is not None:
                    m.bias += dbias

