import collections

import numpy as np

from federatedscope.core.auxiliaries.sampler_builder import get_sampler
from federatedscope.core.auxiliaries.enums import CLIENT_STATE

import logging

logger = logging.getLogger(__name__)


class ClientManager(object):
    def __init__(self,
                 num_client,
                 sample_strategy,
                 id_client_unseen=[]):
        self._num_client_join = 0
        self._num_client_total = num_client

        self._ids_client = []

        # the unseen clients indicate the ones that do not contribute to FL
        # process by training on their local data and uploading their local
        # model update. The splitting is useful to check participation
        # generalization gap in
        # [ICLR'22, What Do We Mean by Generalization in Federated Learning?]
        self._num_client_unseen = len(id_client_unseen)
        self._id_client_unseen = id_client_unseen

        # TODO: achieve by a two-leveled index rather than saving twice
        # Used to maintain the information collected from the clients
        self._info_by_client = collections.defaultdict(dict)
        self._info_by_key = collections.defaultdict(dict)

        # Record the state of the clients (client_id counts from 1)
        self._state_client = dict()

        self.sampler = None
        self.sample_strategy = sample_strategy

    def __assert_client(self, client_id):
        """
        Check if the client_id is legal.
        """
        if client_id in self._ids_client:
            pass
        else:
            raise IndexError(f"Client ID {client_id} doesn't exist.")

    def get_all_client(self):
        return self._ids_client

    @property
    def join_in_client_num(self):
        return self._num_client_join

    def register_client(self, sender):
        """
        Register new client into client manager
        """
        self._num_client_join += 1
        # Assign new id if necessary
        if sender == -1:
            register_id = self._num_client_join
        else:
            register_id = sender

        # Record the client
        self._ids_client.append(register_id)
        self._state_client[register_id] = CLIENT_STATE.IDLE

        return register_id

    def block_unseen_client(self):
        """
        Set the state of the client as CLIENT_STATE.SIDELINE
        """
        if self._num_client_unseen > 0:
            self.change_state(self._num_client_unseen, CLIENT_STATE.SIDELINE)

    def check_client_join_in(self):
        """
        Check if enough clients has joined in.
        """
        return self._num_client_join == self._num_client_total

    def check_client_info(self):
        """
        Check if enough information is collected from the clients
        """
        return len(self._info_by_client) == self._num_client_total

    def update_client_info(self, client_id, info: dict):
        """
        Update client information in the manager.
        """
        if client_id in self._info_by_client:
            logger.info(f"Information of Client #{client_id} is updated by {info}.")
        self._info_by_client.update(info)

        for k, v in info.items():
            self._info_by_key[k][client_id] = v

    def del_client_info(self, client_id):
        """
        Delete the client information.
        """
        if client_id in self._info_by_client:
            del self._info_by_client[client_id]
            for k in self._info_by_client:
                if client_id in self._info_by_client[k]:
                    del self._info_by_client[k][client_id]

    def get_info_by_client(self, client_id):
        """
        Get the client information by client_id
        """
        return self._info_by_client.get(client_id, None)

    def get_info_by_key(self, key):
        """
        Get the client information by key
        """
        return self._info_by_client.get(key, None)

    def change_state(self, indices, state):
        """
        To modify the state of clients (idle or working)
        """
        CLIENT_STATE.assert_value(state)

        if isinstance(indices, list) or isinstance(indices, np.ndarray):
            client_idx = indices
        else:
            client_idx = [indices]

        for client_id in client_idx:
            self._state_client[client_id] = state

    def get_client_by_state(self, state):
        CLIENT_STATE.assert_value(state)

        return [client_id for client_id, client_state in self._state_client.items() if client_state == state]

    def get_idle_client(self):
        """
        Return all the clients with state CLIENT_STATE.IDLE
        """
        return self.get_client_by_state(CLIENT_STATE.IDLE)

    def init_sampler(self):
        """
        Considering the sampling strategy may need client information, we should
        initialize it after finishing client information collection, or
        re-initialize it after updating client information.
        """
        # To sample clients during training
        self.sampler = get_sampler(
            sample_strategy=self.sample_strategy,
            client_num=self._num_client_total,
            client_info=self._info_by_key
        )

    def sample(self, size, perturb=False, change_state=True):
        """Sample idle clients with the specific sampler

        Args:
            size: the number of sample size
            perturb: if perturb the clients before sampling
            change_state: if change the state into CLIENT_STATE.WORKING
        Returns:
            index of the sampled clients
        """
        if self.sampler is None:
            # When we call the function sample, we default that all client information have been collected.
            self.init_sampler()

        # Obtain the idle clients
        clients_idle = self.get_idle_client()
        # Sampling
        clients_sampled = self.sampler.sample(clients_idle, size, perturb)
        if change_state:
            # Change state for the sampled clients
            self.change_state(clients_sampled, CLIENT_STATE.WORKING)
        return clients_sampled
