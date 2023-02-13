import math
from collections import defaultdict

import torch

from federatedscope.core.trainers import BaseTrainer
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer


def copy_params(src):
    tgt = dict()
    for name, t in src.named_parameters():
        if t.requires_grad:
            tgt[name] = t.detach().clone()
    return tgt


def prox_term(cur, last):
    loss = .0
    for name, w in cur.named_parameters():
        loss += 0.5 * torch.sum((w - last[name])**2)
    return loss


def add_noise(model, sigma):
    for p in model.parameters():
        if p.requires_grad:
            p.data += sigma * torch.randn(size=p.shape, device=p.device)


def moving_avg(cur, new, alpha):
    for k, v in cur.items():
        v.data = (1 - alpha) * v + alpha * new[k]


class LocalEntropyTrainer(BaseTrainer):
    def __init__(self, model, data, device, **kwargs):
        # NN modules
        self.model = model
        # FS `ClientData` or your own data
        self.data = data
        # Device name
        self.device = device
        # configs
        self.kwargs = kwargs
        self.config = kwargs['config']
        self.optim_config = self.config.train.optimizer
        self.local_entropy_config = self.config.trainer.local_entropy
        self._thermal = self.local_entropy_config.gamma

    def train(self):
        # Criterion & Optimizer
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        optimizer = get_optimizer(self.model, **self.optim_config)

        # _hook_on_fit_start_init
        self.model.to(self.device)
        current_global_model = copy_params(self.model)
        mu = copy_params(self.model)
        self.model.train()

        num_samples, total_loss = self.run_epoch(optimizer, criterion,
                                                 current_global_model, mu)
        for name, param in self.model.named_parameters():
            if name in mu:
                param.data = mu[name]

        # _hook_on_fit_end
        return num_samples, self.model.cpu().state_dict(), \
            {'loss_total': total_loss, 'avg_loss': total_loss/float(
                num_samples)}

    def run_epoch(self, optimizer, criterion, current_global_model, mu):
        running_loss = 0.0
        num_samples = 0
        # for inputs, targets in self.trainloader:
        for inputs, targets in self.data['train']:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Descent Step
            optimizer.zero_grad()
            outputs = self.model(inputs)
            ce_loss = criterion(outputs, targets)
            loss = ce_loss + self._thermal * prox_term(self.model,
                                                       current_global_model)
            loss.backward()
            optimizer.step()

            # add noise for langevin dynamics
            add_noise(
                self.model,
                math.sqrt(self.optim_config.lr) *
                self.local_entropy_config.eps)

            # acc local updates
            moving_avg(mu, self.model.state_dict(),
                       self.local_entropy_config.alpha)

            with torch.no_grad():
                running_loss += targets.shape[0] * ce_loss.item()

            num_samples += targets.shape[0]
            self._thermal *= self.local_entropy_config.inc_factor

        return num_samples, running_loss

    def evaluate(self, target_data_split_name='test'):
        if target_data_split_name != 'test':
            return {}

        with torch.no_grad():
            criterion = torch.nn.CrossEntropyLoss().to(self.device)

            self.model.to(self.device)
            self.model.eval()
            total_loss = num_samples = num_corrects = 0
            # _hook_on_batch_start_init
            for x, y in self.data[target_data_split_name]:
                # _hook_on_batch_forward
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = criterion(pred, y)
                cor = torch.sum(torch.argmax(pred, dim=-1).eq(y))

                # _hook_on_batch_end
                total_loss += loss.item() * y.shape[0]
                num_samples += y.shape[0]
                num_corrects += cor.item()

            # _hook_on_fit_end
            return {
                f'{target_data_split_name}_acc': float(num_corrects) /
                float(num_samples),
                f'{target_data_split_name}_loss': total_loss,
                f'{target_data_split_name}_total': num_samples,
                f'{target_data_split_name}_avg_loss': total_loss /
                float(num_samples)
            }

    def update(self, model_parameters, strict=False):
        self.model.load_state_dict(model_parameters, strict)

    def get_model_para(self):
        return self.model.cpu().state_dict()
