import numpy as np
import logging

import torch

from federatedscope.core.workers import Server
from federatedscope.core.message import Message
from federatedscope.vertical_fl.Paillier import abstract_paillier
from federatedscope.core.auxiliaries.model_builder import get_model

logger = logging.getLogger(__name__)


class nnServer(Server):
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
        super(nnServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)
        self.model_dict = dict()
        cfg_key_size = config.vertical.key_size
        self.public_key, self.private_key = \
            abstract_paillier.generate_paillier_keypair(n_length=cfg_key_size)
        self.vertical_dims = config.vertical.dims

    def trigger_for_start(self):
        if self.check_client_join_in():
            self.broadcast_client_address()
            self.trigger_for_feat_engr(self.broadcast_model_para)

    def broadcast_model_para(self):
        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[each for each in self.comm_manager.neighbors],
                    state=self.state,
                    content='None'))
