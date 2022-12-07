'''The implementation of ASAM and SAM are borrowed from
    https://github.com/debcaldarola/fedsam
   Caldarola, D., Caputo, B., & Ciccone, M.
   Improving Generalization in Federated Learning by Seeking Flat Minima,
   European Conference on Computer Vision (ECCV) 2022.
'''
from collections import defaultdict
import torch

from federatedscope.core.trainers import BaseTrainer
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer


class ASAM(object):
    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()


class SAM(ASAM):
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()


class SAMTrainer(BaseTrainer):
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
        self.sam_config = self.config.trainer.sam

    def train(self):
        # Criterion & Optimizer
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        optimizer = get_optimizer(self.model, **self.optim_config)

        # _hook_on_fit_start_init
        self.model.to(self.device)
        self.model.train()

        num_samples, total_loss = self.run_epoch(optimizer, criterion)

        # _hook_on_fit_end
        return num_samples, self.model.cpu().state_dict(), \
            {'loss_total': total_loss, 'avg_loss': total_loss/float(
                num_samples)}

    def run_epoch(self, optimizer, criterion):
        if self.sam_config.adaptive:
            minimizer = ASAM(optimizer,
                             self.model,
                             rho=self.sam_config.rho,
                             eta=self.sam_config.eta)
        else:
            minimizer = SAM(optimizer,
                            self.model,
                            rho=self.sam_config.rho,
                            eta=self.sam_config.eta)
        running_loss = 0.0
        num_samples = 0
        # for inputs, targets in self.trainloader:
        for inputs, targets in self.data['train']:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Ascent Step
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            minimizer.ascent_step()

            # Descent Step
            criterion(self.model(inputs), targets).backward()
            minimizer.descent_step()

            with torch.no_grad():
                running_loss += targets.shape[0] * loss.item()

            num_samples += targets.shape[0]

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
