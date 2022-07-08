import numpy as np
import logging

from federatedscope.core.worker import Server
from federatedscope.core.message import Message
from federatedscope.vertical_fl.Paillier import abstract_paillier

logger = logging.getLogger(__name__)


class vFLServer(Server):
    """
    The server class for vertical FL, which customizes the handled
    functions. Please refer to the tutorial for more details about the
    implementation algorithm
    Implementation of Vertical FL refer to `Private federated learning on
    vertically partitioned data via entity resolution and additively
    homomorphic encryption` [Hardy, et al., 2017]
    (https://arxiv.org/abs/1711.10677)
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
        cfg_key_size = config.vertical.key_size
        self.public_key, self.private_key = \
            abstract_paillier.generate_paillier_keypair(n_length=cfg_key_size)
        self.dims = [0] + config.vertical.dims
        self.theta = self.model.state_dict()['fc.weight'].numpy().reshape(-1)
        self.lr = config.train.optimizer.lr

        self.register_handlers('encryped_gradient',
                               self.callback_funcs_for_encryped_gradient)

    def trigger_for_start(self):
        if self.check_client_join_in():
            self.broadcast_public_keys()
            self.broadcast_client_address()
            self.broadcast_model_para()

    def broadcast_public_keys(self):
        self.comm_manager.send(
            Message(msg_type='public_keys',
                    sender=self.ID,
                    receiver=list(self.comm_manager.get_neighbors().keys()),
                    state=self.state,
                    content=self.public_key))

    def broadcast_model_para(self):

        client_ids = self.comm_manager.neighbors.keys()
        cur_idx = 0
        for client_id in client_ids:
            theta_slices = self.theta[cur_idx:cur_idx +
                                      self.dims[int(client_id)]]
            self.comm_manager.send(
                Message(msg_type='model_para',
                        sender=self.ID,
                        receiver=client_id,
                        state=self.state,
                        content=theta_slices))
            cur_idx += self.dims[int(client_id)]

    def callback_funcs_for_encryped_gradient(self, message: Message):
        sample_num, en_v = message.content

        v = np.reshape(
            [self.private_key.decrypt(x) for x in np.reshape(en_v, -1)],
            [sample_num, -1])
        avg_gradients = np.mean(v, axis=0)
        self.theta = self.theta - self.lr * avg_gradients

        self.state += 1
        if self.state % self._cfg.eval.freq == 0 and self.state != \
                self.total_round_num:
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
                role='Global-Eval-Server #',
                forms=self._cfg.eval.report)
            logger.info(formatted_logs)

        if self.state < self.total_round_num:
            # Move to next round of training
            logger.info(f'----------- Starting a new training round (Round '
                        f'#{self.state}) -------------')
            self.broadcast_model_para()
        else:
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

    def evaluate(self):
        test_x = self.data['test']['x']
        test_y = self.data['test']['y']
        loss = np.mean(
            np.log(1 + np.exp(-test_y * np.matmul(test_x, self.theta))))
        acc = np.mean((test_y * np.matmul(test_x, self.theta)) > 0)

        return {'test_loss': loss, 'test_acc': acc, 'test_total': len(test_y)}
