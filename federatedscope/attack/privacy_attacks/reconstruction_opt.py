import torch
import torch.nn.functional as F
from federatedscope.attack.auxiliary.utils import iDLG_trick, total_variation, get_info_diff_loss
import logging

logger = logging.getLogger(__name__)


class DLG(object):
    '''
    Implementation of the paper "Deep Leakage from Gradients": https://papers.nips.cc/paper/2019/file/60a6c4002cc7b29142def8871531281a-Paper.pdf

    References:
        Zhu, Ligeng, Zhijian Liu, and Song Han. "Deep leakage from gradients." Advances in Neural Information Processing Systems 32 (2019).
    '''
    def __init__(self,
                 max_ite,
                 lr,
                 federate_loss_fn,
                 device,
                 federate_method,
                 federate_lr=None,
                 optim='Adam',
                 info_diff_type='l2',
                 is_one_hot_label=False):
        '''

        Args:
            max_ite: the max iteration number; Type int
            lr: learning rate in optimization based reconstruction ; Type float
            federate_loss_fn: The loss function used in FL training; Type object
            device: the device running the reconstruction; Type: str
            federate_method: The federated learning method; Type: str
            federate_lr:The learning rate used in FL training; Type: float; Default None.
            optim: The optimization method used in reconstruction; Type: str; Default: "Adam"
            info_diff_type: The type of loss between the ground-truth gradient/parameter updates info and the reconstructed info; Type: str; Default: "l2"
            is_one_hot_label: whether the label is one-hot; Type: bool; Default: False
        '''

        if federate_method.lower() == "fedavg":
            # check whether the received info is parameter. If yes, the reconstruction attack requires the learning rate of FL
            assert federate_lr is not None

        self.info_is_para = federate_method.lower() == "fedavg"
        self.federate_lr = federate_lr

        self.max_ite = max_ite
        self.lr = lr
        self.device = device
        self.optim = optim
        self.federate_loss_fn = federate_loss_fn
        self.info_diff_type = info_diff_type
        self.info_diff_loss = get_info_diff_loss(info_diff_type)

        self.is_one_hot_label = is_one_hot_label

    def eval(self):
        pass

    def _setup_optimizer(self, parameters):
        if self.optim.lower() == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=self.lr)
        elif self.optim.lower() == 'sgd':  # actually gd
            optimizer = torch.optim.SGD(parameters,
                                        lr=self.lr,
                                        momentum=0.9,
                                        nesterov=True)
        elif self.optim.lower() == 'lbfgs':
            optimizer = torch.optim.LBFGS(parameters)
        else:
            raise ValueError()
        return optimizer

    def _gradient_closure(self, model, optimizer, dummy_data, dummy_label,
                          original_info):
        def closure():
            optimizer.zero_grad()
            model.zero_grad()

            loss = self.federate_loss_fn(
                model(dummy_data),
                dummy_label.view(-1, ).type(torch.LongTensor).to(
                    torch.device(self.device)))

            gradient = torch.autograd.grad(loss,
                                           model.parameters(),
                                           create_graph=True)
            info_diff = 0
            for g_dumby, gt in zip(gradient, original_info):
                info_diff += self.info_diff_loss(g_dumby, gt)
            info_diff.backward()
            return info_diff

        return closure

    def _run_simple_reconstruct(self, model, optimizer, dummy_data, label,
                                original_gradient, closure_fn):

        for ite in range(self.max_ite):
            closure = closure_fn(model, optimizer, dummy_data, label,
                                 original_gradient)
            info_diff = optimizer.step(closure)

            if (ite + 1 == self.max_ite) or ite % 20 == 0:
                logger.info('Ite: {}, gradient difference: {:.4f}'.format(
                    ite, info_diff))
        return dummy_data.detach(), label.detach()

    def get_original_gradient_from_para(self, model, original_info,
                                        model_para_name):
        original_gradient = [
            ((original_para -
              original_info[name].to(torch.device(self.device))) /
             self.federate_lr).detach()
            for original_para, name in zip(model.parameters(), model_para_name)
        ]
        return original_gradient

    def reconstruct(self, model, original_info, data_feature_dim, num_class,
                    batch_size):
        '''

        Args:
            model: The model used in FL; Type: object
            original_info: The message received to perform reconstruction, usually the gradient/parameter updates; Type: list
            data_feature_dim: The feature dimension of dataset; Type: list or Tensor.Size
            num_class: the number of total classes in the dataset; Type: int
            batch_size: the number of samples in the batch that generate the original_info; Type: int

        Returns:
            list of reconstructed data label
            The reconstructed data: Tensor; Size: [batch_size, data_feature_dim]
            The and label: Size: [batch_size]


        '''
        # inital dummy data and label
        dummy_data_dim = [batch_size]
        dummy_data_dim.extend(data_feature_dim)
        dummy_data = torch.randn(dummy_data_dim).to(torch.device(
            self.device)).requires_grad_(True)

        para_trainable_name = []
        for p in model.named_parameters():
            para_trainable_name.append(p[0])

        if self.info_is_para:
            original_gradient = self.get_original_gradient_from_para(
                model, original_info, model_para_name=para_trainable_name)
        else:
            original_gradient = [
                grad.to(torch.device(self.device)) for k, grad in original_info
            ]

        label = iDLG_trick(original_gradient,
                           num_class=num_class,
                           is_one_hot_label=self.is_one_hot_label)
        label = label.to(torch.device(self.device))

        # setup optimizer
        optimizer = self._setup_optimizer([dummy_data])

        self._run_simple_reconstruct(model,
                                     optimizer,
                                     dummy_data,
                                     label=label,
                                     original_gradient=original_gradient,
                                     closure_fn=self._gradient_closure)

        return dummy_data.detach(), label.detach()


class InvertGradient(DLG):
    '''
    The implementation of "Inverting Gradients - How easy is it to break privacy in federated learning?".
    Link: https://proceedings.neurips.cc/paper/2020/hash/c4ede56bbd98819ae6112b20ac6bf145-Abstract.html

    References:
        Geiping, Jonas, et al. "Inverting gradients-how easy is it to break privacy in federated learning?." Advances in Neural Information Processing Systems 33 (2020): 16937-16947.
    '''
    def __init__(self,
                 max_ite,
                 lr,
                 federate_loss_fn,
                 device,
                 federate_method,
                 federate_lr=None,
                 alpha_TV=0.001,
                 info_diff_type='sim',
                 optim='Adam'):
        super(InvertGradient, self).__init__(max_ite,
                                             lr,
                                             federate_loss_fn,
                                             device,
                                             federate_method,
                                             federate_lr=federate_lr,
                                             optim=optim,
                                             info_diff_type=info_diff_type)
        self.alpha_TV = alpha_TV
        if self.info_diff_type != 'sim':
            logger.info(
                'Force the info_diff_type to be cosine similarity loss in InvertGradient attack method!'
            )
            self.info_diff_type = 'sim'
            self.info_diff_loss = get_info_diff_loss(self.info_diff_type)

    def _gradient_closure(self, model, optimizer, dummy_data, dummy_label,
                          original_gradient):
        def closure():
            optimizer.zero_grad()
            model.zero_grad()
            loss = self.federate_loss_fn(
                model(dummy_data),
                dummy_label.view(-1, ).type(torch.LongTensor).to(
                    torch.device(self.device)))

            gradient = torch.autograd.grad(loss,
                                           model.parameters(),
                                           create_graph=True)
            gradient_diff = 0

            for g_dummy, gt in zip(gradient, original_gradient):
                gradient_diff += self.info_diff_loss(g_dummy, gt)

            # add total variance regularization
            if self.alpha_TV > 0:
                gradient_diff += self.alpha_TV * total_variation(dummy_data)
            gradient_diff.backward()
            return gradient_diff

        return closure
