import logging

import yaml

import numpy as np

from federatedscope.core.message import Message
from federatedscope.core.worker import Server
from federatedscope.autotune import Continuous, Discrete, split_raw_config
from federatedscope.autotune.algos import random_search

logger = logging.getLogger(__name__)


class FedExServer(Server):
    """Some code snippets are borrowed from the open-sourced FedEx (https://github.com/mkhodak/FedEx)
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

        # initialize action space and the policy
        with open(config.hpo.fedex.ss, 'r') as ips:
            ss = yaml.load(ips, Loader=yaml.FullLoader)
        _, tbd_config = split_raw_config(ss)
        self._cfsp = random_search(tbd_config, config.hpo.fedex.num_arms)
        sizes = [len(self._cfsp)]
        self._z = [np.full(size, -np.log(size)) for size in sizes]
        self._theta = [np.exp(z) for z in self._z]

        super(FedExServer, self).__init__(ID, state, config, data, model, client_num, total_round_num, device, strategy, **kwargs)

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1):
        """
        To broadcast the message to all clients or sampled clients
        """
        if sample_client_num > 0:
            receiver = np.random.choice(np.arange(1, self.client_num + 1),
                                        size=sample_client_num,
                                        replace=False).tolist()
        else:
            # broadcast to all clients
            receiver = list(self.comm_manager.neighbors.keys())

        if self._noise_injector is not None and msg_type == 'model_para':
            # Inject noise only when broadcast parameters
            for model_idx_i in range(len(self.models)):
                num_sample_clients = [
                    v["num_sample"] for v in self.join_in_info.values()
                ]
                self._noise_injector(self._cfg, num_sample_clients,
                                     self.models[model_idx_i])

        if self.model_num > 1:
            model_para = [model.state_dict() for model in self.models]
        else:
            model_para = self.model.state_dict()

        # sample the hyper-parameter config specific to the clients
        
        for rcv_idx in receiver:
            cfg_idx = [np.random.choice(len(theta), p=theta) for theta in self._theta]
            sampled_cfg = self._cfsp[cfg_idx[0]]
            content = {'model_param': model_para, 'hyperparam': sampled_cfg}
            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=[rcv_idx],
                        state=self.state,
                        content=content))
        if self._cfg.federate.online_aggr:
            for idx in range(self.model_num):
                self.aggregators[idx].reset()
