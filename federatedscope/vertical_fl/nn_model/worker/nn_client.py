import numpy as np
import logging

import torch
from torch import nn
from torch.autograd import Variable
from sklearn import metrics

from federatedscope.core.workers import Client
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)


class nnClient(Client):
    """

    """
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 *args,
                 **kwargs):

        super(nnClient,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, *args, **kwargs)

        self.middle_result_dict = dict()
        self.data = data
        self.client_num = config.federate.client_num
        self.own_label = ('y' in data['train'])
        self.dims = [0] + config.vertical.dims
        self.bottom_input_layer = self.dims[self.ID] - self.dims[self.ID - 1]
        self.bottom_output_layer = config.vertical.output_layer[self.ID - 1]
        self.top_input_layer = None
        if self.own_label:
            self.top_input_layer = np.sum(config.vertical.output_layer)
        self.eval_middle_result_dict = dict()

        self._init_data_related_var()

        self.register_handlers('model_para',
                               self.callback_funcs_for_model_para)
        self.register_handlers('data_sample',
                               self.callback_func_for_data_sample)
        self.register_handlers('middle_result',
                               self.callback_func_for_middle_result)
        self.register_handlers('grad', self.callback_func_for_grad)
        self.register_handlers('continue', self.callback_funcs_for_continue)
        self.register_handlers('eval', self.callback_func_for_eval)
        self.register_handlers('eval_middle_result',
                               self.callback_func_for_eval_middle_result)

    # def train(self):
    #     raise NotImplementedError
    #
    # def eval(self):
    #     raise NotImplementedError

    def _init_data_related_var(self):
        self.trainer._init_for_train(self.bottom_input_layer,
                                     self.bottom_output_layer,
                                     self.top_input_layer)
        self.test_x = torch.Tensor(self.data['test']['x'])

        if self.own_label:
            self.test_y = np.vstack(self.data['test']['y']).reshape(-1, 1)
            self.test_y = torch.Tensor(self.test_y)

    def callback_funcs_for_model_para(self, message: Message):
        self.state = message.state

        if self.own_label:
            self.start_a_new_training_round()

    def start_a_new_training_round(self):
        logger.info(f'----------- Starting a new round (Round '
                    f'#{self.state}) -------------')
        batch_index = self.trainer.fetch_train_data()

        receiver = [
            each for each in list(self.comm_manager.neighbors.keys())
            if each != self.server_id
        ]
        send_message = Message(msg_type='data_sample',
                               sender=self.ID,
                               state=self.state,
                               receiver=receiver,
                               content=batch_index)
        self.comm_manager.send(send_message)
        self.train()

    def callback_func_for_data_sample(self, message: Message):
        self.state = message.state
        batch_index = message.content
        _ = self.trainer.fetch_train_data(index=batch_index)
        self.train()

    def train(self):
        middle_result = self.trainer.train_bottom()
        if self.own_label:
            self.middle_result_dict[self.ID] = middle_result
        else:
            self.comm_manager.send(
                Message(msg_type='middle_result',
                        sender=self.ID,
                        receiver=[self.client_num],
                        state=self.state,
                        content=middle_result))

    def callback_func_for_middle_result(self, message: Message):
        middle_result = message.content
        self.middle_result_dict[message.sender] = middle_result
        if len(self.middle_result_dict) == self.client_num:
            client_ids = list(self.middle_result_dict.keys())
            client_ids = sorted(client_ids)

            middle_result = torch.cat(
                [self.middle_result_dict[i] for i in client_ids], 1)
            self.middle_result_dict = dict()

            middle_result = Variable(middle_result, requires_grad=True)

            train_loss, grad_list = self.trainer.train_top(middle_result)
            # print(train_loss)
            for i in range(self.client_num - 1):
                self.comm_manager.send(
                    Message(msg_type='grad',
                            sender=self.ID,
                            receiver=[i + 1],
                            state=self.state,
                            content=grad_list[i]))

    def callback_func_for_grad(self, message: Message):
        grad = message.content
        # print("==========")
        # print(grad)
        # # grad = torch.from_numpy(grad)
        # print('-----')
        self.trainer.bottom_model_backward(grad)
        # print(self.a)
        # print(grad)
        # input()
        # self.a.backward(grad)
        # print(self.bottom_model[0].weight.grad)
        # self.bottom_model_opt.step()

        self.comm_manager.send(
            Message(msg_type='continue',
                    sender=self.ID,
                    receiver=[self.client_num],
                    state=self.state,
                    content='None'))

    def callback_funcs_for_continue(self, message: Message):

        if (self.state+1) % self._cfg.eval.freq == 0 and \
                (self.state+1) != self._cfg.federate.total_round_num:
            self.eval_middle_result_dict[self.ID] = self.trainer.bottom_model(
                self.test_x)
            self.comm_manager.send(
                Message(msg_type='eval',
                        sender=self.ID,
                        receiver=[
                            each for each in self.comm_manager.neighbors
                            if each != self.server_id
                        ],
                        state=self.state,
                        content='None'))

        elif self.state + 1 < self._cfg.federate.total_round_num:
            self.state += 1
            self.start_a_new_training_round()
        else:
            self.comm_manager.send(
                Message(msg_type='eval',
                        sender=self.ID,
                        receiver=[
                            each for each in self.comm_manager.neighbors
                            if each != self.server_id
                        ],
                        state=self.state,
                        content='None'))
            self.eval_middle_result_dict[self.ID] = self.trainer.bottom_model(
                self.test_x)

    def callback_func_for_eval(self, message: Message):
        eval_middle_result = self.trainer.bottom_model(self.test_x)
        self.comm_manager.send(
            Message(msg_type='eval_middle_result',
                    sender=self.ID,
                    receiver=[message.sender],
                    state=self.state,
                    content=eval_middle_result))

    def callback_func_for_eval_middle_result(self, message: Message):
        eval_middle_result = message.content
        self.eval_middle_result_dict[message.sender] = eval_middle_result
        if len(self.eval_middle_result_dict) == self.client_num:
            client_ids = list(self.eval_middle_result_dict.keys())
            client_ids = sorted(client_ids)

            eval_middle_result = torch.cat(
                [self.eval_middle_result_dict[i] for i in client_ids], 1)
            self.eval_middle_result_dict = dict()

            y_hat = self.trainer.top_model(eval_middle_result)

            test_loss = self.trainer.criterion(y_hat, self.test_y)

            auc = metrics.roc_auc_score(
                self.test_y.reshape(-1).detach().numpy(),
                y_hat.reshape(-1).detach().numpy())
            y_hat = (y_hat >= 0.5)

            acc = torch.sum(y_hat == self.test_y) / len(self.test_y)

            self.metrics = {
                'test_loss': test_loss.detach().numpy(),
                "test_auc": auc,
                "test_acc": acc.numpy(),
                'test_total': len(self.test_y)
            }

            self._monitor.update_best_result(self.best_results,
                                             self.metrics,
                                             results_type='server_global_eval')

            formatted_logs = self._monitor.format_eval_res(
                self.metrics,
                rnd=self.state,
                role='Server #',
                forms=self._cfg.eval.report)

            logger.info(formatted_logs)

            # print("test loss:",
            #       test_loss.detach().numpy(), "test auc:", auc, "test acc:",
            #       acc.numpy())
            if self.state + 1 < self._cfg.federate.total_round_num:
                self.state += 1
                self.start_a_new_training_round()
