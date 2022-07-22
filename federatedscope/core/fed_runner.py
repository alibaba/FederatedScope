import logging

from collections import deque

import numpy as np

from federatedscope.core.worker import Server, Client
from federatedscope.core.gpu_manager import GPUManager
from federatedscope.core.auxiliaries.model_builder import get_model

logger = logging.getLogger(__name__)


class FedRunner(object):
    """
    This class is used to construct an FL course, which includes `_set_up`
    and `run`.

    Arguments:
        data: The data used in the FL courses, which are formatted as {
        'ID':data} for standalone mode. More details can be found in
        federatedscope.core.auxiliaries.data_builder .
        server_class: The server class is used for instantiating a (
        customized) server.
        client_class: The client class is used for instantiating a (
        customized) client.
        config: The configurations of the FL course.
        client_config: The clients' configurations.
    """
    def __init__(self,
                 data,
                 server_class=Server,
                 client_class=Client,
                 config=None,
                 client_config=None):
        self.data = data
        self.server_class = server_class
        self.client_class = client_class
        self.cfg = config
        self.client_cfg = client_config

        self.mode = self.cfg.federate.mode.lower()
        self.gpu_manager = GPUManager(gpu_available=self.cfg.use_gpu,
                                      specified_device=self.cfg.device)

        self.unseen_clients_id = []
        if self.cfg.federate.unseen_clients_rate > 0:
            self.unseen_clients_id = np.random.choice(
                np.arange(1, self.cfg.federate.client_num + 1),
                size=max(
                    1,
                    int(self.cfg.federate.unseen_clients_rate *
                        self.cfg.federate.client_num)),
                replace=False).tolist()

        if self.mode == 'standalone':
            self.shared_comm_queue = deque()
            self._setup_for_standalone()
            # in standalone mode, by default, we print the trainer info only
            # once for better logs readability
            trainer_representative = self.client[1].trainer
            if trainer_representative is not None:
                trainer_representative.print_trainer_meta_info()
        elif self.mode == 'distributed':
            self._setup_for_distributed()

    def _setup_for_standalone(self):
        """
        To set up server and client for standalone mode.
        """

        if self.cfg.backend == 'torch':
            import torch
            torch.set_num_threads(1)

        assert self.cfg.federate.client_num != 0, \
            "In standalone mode, self.cfg.federate.client_num should be " \
            "non-zero. " \
            "This is usually cased by using synthetic data and users not " \
            "specify a non-zero value for client_num"

        if self.cfg.federate.method == "global":
            if self.cfg.federate.client_num != 1:
                from federatedscope.core.auxiliaries.data_builder import \
                    merge_data
                if self.cfg.data.server_holds_all:
                    assert self.data[0] is not None \
                        and len(self.data[0]) != 0, \
                        "You specified cfg.data.server_holds_all=True " \
                        "but data[0] is None. Please check whether you " \
                        "pre-process the data[0] correctly"
                    self.data[1] = self.data[0]
                else:
                    logger.info(f"Will merge data from clients whose ids in "
                                f"[1, {self.cfg.federate.client_num}]")
                    self.data[1] = merge_data(
                        all_data=self.data,
                        merged_max_data_id=self.cfg.federate.client_num)
                self.cfg.defrost()
                self.cfg.federate.client_num = 1
                self.cfg.federate.sample_client_num = 1
                self.cfg.freeze()

        self.server = self._setup_server()

        self.client = dict()

        # assume the client-wise data are consistent in their input&output
        # shape
        self._shared_client_model = get_model(
            self.cfg.model, self.data[1], backend=self.cfg.backend
        ) if self.cfg.federate.share_local_model else None

        for client_id in range(1, self.cfg.federate.client_num + 1):
            self.client[client_id] = self._setup_client(
                client_id=client_id, client_model=self._shared_client_model)

    def _setup_for_distributed(self):
        """
        To set up server or client for distributed mode.
        """
        self.server_address = {
            'host': self.cfg.distribute.server_host,
            'port': self.cfg.distribute.server_port
        }
        if self.cfg.distribute.role == 'server':
            self.server = self._setup_server()
        elif self.cfg.distribute.role == 'client':
            # When we set up the client in the distributed mode, we assume
            # the server has been set up and number with #0
            self.client_address = {
                'host': self.cfg.distribute.client_host,
                'port': self.cfg.distribute.client_port
            }
            self.client = self._setup_client()

    def run(self):
        """
        To run an FL course, which is called after server/client has been
        set up.
        For the standalone mode, a shared message queue will be set up to
        simulate ``receiving message``.
        """
        if self.mode == 'standalone':
            # trigger the FL course
            for each_client in self.client:
                self.client[each_client].join_in()

            if self.cfg.federate.online_aggr:
                # any broadcast operation would be executed client-by-client
                # to avoid the existence of #clients messages at the same time.
                # currently, only consider centralized topology
                def is_broadcast(msg):
                    return len(msg.receiver) >= 1 and msg.sender == 0

                cached_bc_msgs = []
                cur_idx = 0
                while True:
                    if len(self.shared_comm_queue) > 0:
                        msg = self.shared_comm_queue.popleft()
                        if is_broadcast(msg):
                            cached_bc_msgs.append(msg)
                            # assume there is at least one client
                            msg = cached_bc_msgs[0]
                            self._handle_msg(msg, rcv=msg.receiver[cur_idx])
                            cur_idx += 1
                            if cur_idx >= len(msg.receiver):
                                del cached_bc_msgs[0]
                                cur_idx = 0
                        else:
                            self._handle_msg(msg)
                    elif len(cached_bc_msgs) > 0:
                        msg = cached_bc_msgs[0]
                        self._handle_msg(msg, rcv=msg.receiver[cur_idx])
                        cur_idx += 1
                        if cur_idx >= len(msg.receiver):
                            del cached_bc_msgs[0]
                            cur_idx = 0
                    else:
                        # finished
                        break

            else:
                while len(self.shared_comm_queue) > 0:
                    msg = self.shared_comm_queue.popleft()
                    self._handle_msg(msg)

            self.server._monitor.finish_fed_runner(fl_mode=self.mode)

            return self.server.best_results

        elif self.mode == 'distributed':
            if self.cfg.distribute.role == 'server':
                self.server.run()
                return self.server.best_results
            elif self.cfg.distribute.role == 'client':
                self.client.join_in()
                self.client.run()

    def _setup_server(self):
        """
        Set up the server
        """
        self.server_id = 0
        if self.mode == 'standalone':
            if self.server_id in self.data:
                server_data = self.data[self.server_id]
                model = get_model(self.cfg.model,
                                  server_data,
                                  backend=self.cfg.backend)
            else:
                server_data = None
                model = get_model(
                    self.cfg.model, self.data[1], backend=self.cfg.backend
                )  # get the model according to client's data if the server
                # does not own data
            kw = {'shared_comm_queue': self.shared_comm_queue}
        elif self.mode == 'distributed':
            server_data = self.data
            model = get_model(self.cfg.model,
                              server_data,
                              backend=self.cfg.backend)
            kw = self.server_address
        else:
            raise ValueError('Mode {} is not provided'.format(
                self.cfg.mode.type))

        if self.server_class:
            self._server_device = self.gpu_manager.auto_choice()
            server = self.server_class(
                ID=self.server_id,
                config=self.cfg,
                data=server_data,
                model=model,
                client_num=self.cfg.federate.client_num,
                total_round_num=self.cfg.federate.total_round_num,
                device=self._server_device,
                unseen_clients_id=self.unseen_clients_id,
                **kw)

            if self.cfg.nbafl.use:
                from federatedscope.core.trainers.trainer_nbafl import \
                    wrap_nbafl_server
                wrap_nbafl_server(server)

        else:
            raise ValueError

        logger.info('Server #{:d} has been set up ... '.format(self.server_id))

        return server

    def _setup_client(
        self,
        client_id=-1,
        client_model=None,
    ):
        """
        Set up the client
        """
        self.server_id = 0
        if self.mode == 'standalone':
            client_data = self.data[client_id]
            kw = {'shared_comm_queue': self.shared_comm_queue}
        elif self.mode == 'distributed':
            client_data = self.data
            kw = self.client_address
            kw['server_host'] = self.server_address['host']
            kw['server_port'] = self.server_address['port']
        else:
            raise ValueError('Mode {} is not provided'.format(
                self.cfg.mode.type))

        if self.client_class:
            client_specific_config = self.cfg.clone()
            if self.client_cfg:
                client_specific_config.defrost()
                client_specific_config.merge_from_other_cfg(
                    self.client_cfg.get('client_{}'.format(client_id)))
                client_specific_config.freeze()
            client_device = self._server_device if \
                self.cfg.federate.share_local_model else \
                self.gpu_manager.auto_choice()
            client = self.client_class(
                ID=client_id,
                server_id=self.server_id,
                config=client_specific_config,
                data=client_data,
                model=client_model or get_model(client_specific_config.model,
                                                client_data,
                                                backend=self.cfg.backend),
                device=client_device,
                is_unseen_client=client_id in self.unseen_clients_id,
                **kw)
        else:
            raise ValueError

        if client_id == -1:
            logger.info('Client (address {}:{}) has been set up ... '.format(
                self.client_address['host'], self.client_address['port']))
        else:
            logger.info(f'Client {client_id} has been set up ... ')

        return client

    def _handle_msg(self, msg, rcv=-1):
        """
        To simulate the message handling process (used only for the
        standalone mode)
        """
        if rcv != -1:
            # simulate broadcast one-by-one
            self.client[rcv].msg_handlers[msg.msg_type](msg)
            return

        _, receiver = msg.sender, msg.receiver
        download_bytes, upload_bytes = msg.count_bytes()
        if not isinstance(receiver, list):
            receiver = [receiver]
        for each_receiver in receiver:
            if each_receiver == 0:
                self.server.msg_handlers[msg.msg_type](msg)
                self.server._monitor.track_download_bytes(download_bytes)
            else:
                self.client[each_receiver].msg_handlers[msg.msg_type](msg)
                self.client[each_receiver]._monitor.track_download_bytes(
                    download_bytes)
