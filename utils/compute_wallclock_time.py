import torch
import time
from models import *


def compute_wallclock_time(net, input_res, batch_size):
    with torch.no_grad():
        x = torch.cuda.FloatTensor(batch_size, 3, input_res, input_res).normal_()
        net = net.cuda()
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        a = time.perf_counter()
        out = net(x)
        torch.cuda.synchronize() # wait for forward to finish
        b = time.perf_counter()
        print('batch GPU {:.02e}s'.format(b - a))
    return b-a
