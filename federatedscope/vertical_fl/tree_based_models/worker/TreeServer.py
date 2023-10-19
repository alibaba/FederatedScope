import numpy as np

from federatedscope.core.workers import Server
from federatedscope.core.message import Message
from federatedscope.core.secret_sharing import MultiplicativeSecretSharing

import logging

logger = logging.getLogger(__name__)


class TreeServer(Server):
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=2,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        super(TreeServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)

        self.batch_size = self._cfg.dataloader.batch_size
        self.feature_partition = np.diff(self._cfg.vertical.dims, prepend=0)
        self.total_num_of_feature = self._cfg.vertical.dims[-1]
        self._init_data_related_var()

        if self._cfg.vertical.eval_protection == 'ss':
            self.ss = MultiplicativeSecretSharing(
                shared_party_num=self.client_num)

    def _init_data_related_var(self):
        pass

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        # The server broadcasts the order to trigger the training process
        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=list(self.comm_manager.get_neighbors().keys()),
                    state=self.state,
                    content='None'))

    def terminate(self, msg_type='finish'):
        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=list(self.comm_manager.get_neighbors().keys()),
                    state=self.state,
                    content='None'))
        # jump out running
        self.state = self.total_round_num + 1
