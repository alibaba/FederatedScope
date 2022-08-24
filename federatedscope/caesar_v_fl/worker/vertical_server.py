import numpy as np
import logging

from federatedscope.core.worker import Server
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)


class vFLServer(Server):
    """
    The server class for vertical FL, which customizes the handled
    functions. Please refer to the tutorial for more details about the
    implementation algorithm
    Implementation of Vertical FL refer to `When Homomorphic Encryption
    Marries Secret Sharing: Secure Large-Scale Sparse Logistic Regression and
    Applications in Risk Control` [Chen, et al., 2021]
    (https://arxiv.org/abs/2008.08753)
    """
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        super(vFLServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)
        self.dims = [0] + config.caesar_vertical.dims
        self.omega = self.model.state_dict()['fc.weight'].numpy().reshape(-1)
        self.lr = config.train.optimizer.lr
        self.w_a = None
        self.w_b = None
        self.coll = dict()

        self.register_handlers('a_para_for_server', self.callback_func_for_a)
        self.register_handlers('b_para_for_server', self.callback_func_for_b)

    def trigger_for_start(self):
        if self.check_client_join_in():
            self.broadcast_client_address()
            self.broadcast_model_para()

    def broadcast_model_para(self):
        client_ids = self.comm_manager.neighbors.keys()
        cur_idx = 0
        for client_id in client_ids:
            omega_slices = self.omega[cur_idx:cur_idx +
                                      self.dims[int(client_id)]]
            self.comm_manager.send(
                Message(msg_type='model_para',
                        sender=self.ID,
                        receiver=client_id,
                        state=self.state,
                        content=(self.total_round_num, omega_slices)))
            cur_idx += self.dims[int(client_id)]

    def callback_func_for_a(self, message: Message):
        self.w_a = message.content
        self.coll['wa'] = self.w_a
        self.output()

    def callback_func_for_b(self, message: Message):
        self.w_b = message.content
        self.coll['wb'] = self.w_b
        self.output()

    def output(self):
        if len(self.coll) == 2:
            metrics = self.evaluate()
            self._monitor.update_best_result(
                self.best_results,
                metrics,
                results_type='server_global_eval',
                round_wise_update_key=self._cfg.eval.
                best_res_update_round_wise_key)
            formatted_logs = self._monitor.format_eval_res(
                metrics,
                rnd=self.state,
                role='Server #',
                forms=self._cfg.eval.report)
            logger.info(formatted_logs)
            self.coll = dict()

    def evaluate(self):
        self.omega = np.concatenate([self.coll['wa'], self.coll['wb']],
                                    axis=-1)
        self.coll = dict()
        test_x = self.data['test']['x']
        test_y = self.data['test']['y']
        loss = np.mean(
            np.log(1 + np.exp(-test_y * np.matmul(test_x, self.omega))))
        acc = np.mean((test_y * np.matmul(test_x, self.omega)) > 0)
        return {'test_loss': loss, 'test_acc': acc, 'test_total': len(test_y)}
