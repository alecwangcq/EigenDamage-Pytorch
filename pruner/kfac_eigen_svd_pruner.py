import numpy as np
import time
import torch
import torch.nn as nn
from sktensor import dtensor, cp_als
from pruner.kfac_eigen_pruner import KFACEigenPruner


def get_UDV_decomposition(W, method='svd'):
    # current implementation is svd
    c_out, khkw, c_in = W.shape
    method = method.lower()
    with torch.no_grad():
        if method == 'svd':
            m_W = W.mean(1)
            U, _, V = torch.svd(m_W)
            D = []
            for r in range(khkw):
                W_r = W[:, r, :]  # c_out * c_in  ->
                c = min(c_out, c_in)
                D_w = torch.diag(U.t() @ W_r @ V).view(c, 1)
                D.append(D_w)
            S = torch.stack(D, dim=1)
        elif method == 'svd_avg':
            pass
        elif method == 'als':
            # m = min(c_out, c_in)
            # U: c_out * m
            # S: k^2 * m
            # V: c_in * m
            rank = min(c_out, c_in)

            tic = time.clock()
            T = dtensor(W.data.cpu().numpy())
            P, fit, itr, exectimes = cp_als(T, rank, init='random')
            U = np.array(P.U[0])  # c_out * rank
            S = np.array(P.U[1]).T  # k^2 * rank --> rank * k^2
            V = np.array(P.U[2] * P.lmbda)  # c_in * rank
            print('CP decomposition done. It cost %.5f secs. fit: %f' % (time.clock() - tic, fit[0]))
            V = torch.FloatTensor(V).cuda()
            S = torch.FloatTensor(S).cuda()
            U = torch.FloatTensor(U).cuda()

        else:
            raise NotImplementedError("Method {} not supported!".format(method))

    return U, S, V


class KFACEigenSVDPruner(KFACEigenPruner):

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
        super(KFACEigenSVDPruner, self).__init__(model,
                                                 builder,
                                                 config,
                                                 writer,
                                                 logger,
                                                 prune_ratio_limit,
                                                 batch_averaged,
                                                 fix_rotation,
                                                 use_patch,
                                                 fix_layers)

    def make_pruned_model(self, dataloader, criterion, device, fisher_type, prune_ratio, normalize=True, re_init=False):
        self._prepare_model()
        self.init_step()

        self._compute_fisher(dataloader, criterion, device, fisher_type)
        self._update_inv()  # eigen decomposition of fisher

        self._get_unit_importance(normalize)
        self._merge_Qs()  # update the eigen basis

        self._do_prune(prune_ratio, all_keep=False)
        self._make_depth_separable()

        self._rm_hooks()
        self._build_pruned_model(re_init)
        print(self.model)
        self._clear_buffer()


    def _make_depth_separable(self):
        assert self._inversed
        for idx, m in enumerate(self.remain_modules):
            if isinstance(m, nn.Conv2d):
                W_star = self.W_star[m]  # c_out * (kh * kw) * c_in
                m.groups = min(W_star.shape[-1], W_star.shape[0])
                U, W_sep, V = get_UDV_decomposition(W_star, method='als')
                try:
                    self.Q_a[m] = self.Q_a[m] @ V
                    self.Q_g[m] = self.Q_g[m] @ U
                    self.W_star[m] = W_sep
                except:
                    import pdb; pdb.set_trace()




        # self.remain_modules = []
        # for m in self.modules:
        #     self.remain_modules.append(m)


