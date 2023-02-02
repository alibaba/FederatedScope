import torch
from torch.optim.optimizer import Optimizer
import numpy as np

class ADADP(Optimizer):

    def __init__(self, params, lr=1e-3, tol=1.0):

        defaults = dict(lr=lr)

        self.p0 = None
        self.p1 = None
        self.lrs = lr
        self.tol = tol
        self.accepted = 0
        self.failed = 0

        self.lrs_history = []
        self._steps = 0
        super(ADADP, self).__init__(params, defaults)

    def step(self):
        if self._steps % 2 == 0:
            self._step1()
        else:
            self._step2()
        self._steps += 1

    def _step1(self):

        del self.p0
        self.p0 = []

        del self.p1
        self.p1 = []

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue

                dd = p.data.clone()
                self.p0.append(dd)

                self.p1.append(p.data  -  self.lrs*p.grad.data)
                p.data.add_(-0.5*self.lrs, p.grad.data)

    def _step2(self):

        for group in self.param_groups:

            err_e = 0.0

            for ijk, p in enumerate(group['params']):
                p.data.add_(-0.5*self.lrs, p.grad.data)
                _p1 = self.p1[ijk]
                err_e += (((_p1 - p.data) ** 2 /
                           (torch.max(torch.ones_like(_p1),
                                      _p1 ** 2))).norm(1))

            err_e = np.sqrt(float(err_e))

            self.lrs = float(self.lrs*min(max(np.sqrt(self.tol/err_e),0.9), 1.1))

            ## Accept the step only if err < self.tol.
            ## Can be sometimes neglected (more accepted steps)
            if err_e > 1.0*self.tol:
               for ijk,p in enumerate(group['params']):
                   p.data = self.p0[ijk]
            if err_e < self.tol:
               self.accepted += 1
            else :
               self.failed += 1

            self.lrs_history.append(self.lrs)