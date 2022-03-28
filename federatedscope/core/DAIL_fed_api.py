import logging

from collections import deque

from federatedscope.core.worker import Server, Client
from federatedscope.core.gpu_manager import GPUManager
from federatedscope.core.auxiliaries.model_builder import get_model


class DAILFed(object):
    def __init__(self,
                 data,
                 server_class=Server,
                 client_class=Client,
                 config=None):
        self.data = data
        self.server_class = server_class
        self.client_class = client_class
        self.cfg = config

        self.mode = self.cfg.federate.mode.lower()
        self.gpu_manager = GPUManager(gpu_available=self.cfg.use_gpu,
                                      specified_device=self.cfg.device)

        if self.mode == 'standalone':
            self.shared_comm_queue = deque()
            self._setup_for_standalone()
        elif self.mode == 'distributed':
            self._setup_for_distributed()

    def _setup_for_standalone(self):
        self.server = self._setup_server()

        self.client = dict()
        assert self.cfg.federate.client_num != 0, \
            "In standalone mode, self.cfg.federate.client_num should be non-zero. " \
            "This is usually cased by using synthetic data and users not specify a non-zero value for client_num"

        # assume the client-wise data are consistent in their input&output shape
        self._shared_client_model = get_model(
            self.cfg.model,
            self.data[1]) if self.cfg.federate.share_local_model else None
        for client_id in range(1, self.cfg.federate.client_num + 1):
            self.client[client_id] = self._setup_client(
                client_id=client_id, client_model=self._shared_client_model)

    def _setup_for_distributed(self):
        self.server_address = {
            'host': self.cfg.distribute.server_host,
            'port': self.cfg.distribute.server_port
        }
        if self.cfg.distribute.role == 'server':
            self.server = self._setup_server()
        elif self.cfg.distribute.role == 'client':
            # When we set up the client in the distributed mode, we assume the server has been set up and number with #0
            self.client_address = {
                'host': self.cfg.distribute.client_host,
                'port': self.cfg.distribute.client_port
            }
            self.client = self._setup_client()

    def run(self):
        if self.mode == 'standalone':
            # trigger the FL course
            for each_client in self.client:
                self.client[each_client].join_in()

            if self.cfg.federate.online_aggr:
                # any broadcast operation would be executed client-by-client to avoid the existence of #clients messages at the same time.
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
            return self.server.best_results

        elif self.mode == 'distributed':
            if self.cfg.distribute.role == 'server':
                self.server.run()
                return self.server.best_results
            elif self.cfg.distribute.role == 'client':
                self.client.join_in()
                self.client.run()

    def _setup_server(self):
        self.server_id = 0
        if self.mode == 'standalone':
            if self.server_id in self.data:
                server_data = self.data[self.server_id]
                model = get_model(self.cfg.model, server_data)
            else:
                server_data = None
                model = get_model(
                    self.cfg.model, self.data[1]
                )  # get the model according to client's data if the server does not own data
            kw = {'shared_comm_queue': self.shared_comm_queue}
        elif self.mode == 'distributed':
            server_data = self.data
            model = get_model(self.cfg.model,
                              server_data)
            kw = self.server_address
        else:
            raise ValueError('Mode {} is not provided'.format(
                self.cfg.mode.type))

        if self.server_class:
            server = self.server_class(
                ID=self.server_id,
                config=self.cfg,
                data=server_data,
                model=model,
                client_num=self.cfg.federate.client_num,
                total_round_num=self.cfg.federate.total_round_num,
                device=self.gpu_manager.auto_choice(),
                **kw)

            if self.cfg.nbafl.use:
                from federatedscope.core.trainers.trainer_nbafl import wrap_nbafl_server
                wrap_nbafl_server(server)

        else:
            raise ValueError

        logging.info('Server #{:d} has been set up ... '.format(
            self.server_id))

        return server

    def _setup_client(self, client_id=-1, client_model=None):
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
            client = self.client_class(
                ID=client_id,
                server_id=self.server_id,
                config=client_specific_config,
                data=client_data,
                model=client_model or get_model(self.cfg.model, client_data),
                device=self.gpu_manager.auto_choice(),
                **kw)
        else:
            raise ValueError

        if client_id == -1:
            logging.info('Client (address {}:{}) has been set up ... '.format(
                self.client_address['host'], self.client_address['port']))
        else:
            logging.info('Client has been set up ... '.format(client_id))

        return client

    def _handle_msg(self, msg, rcv=-1):
        if rcv != -1:
            # simulate broadcast one-by-one
            self.client[rcv].msg_handlers[msg.msg_type](msg)
            return

        sender, receiver = msg.sender, msg.receiver
        if not isinstance(receiver, list):
            receiver = [receiver]
        for each_receiver in receiver:
            if each_receiver == 0:
                self.server.msg_handlers[msg.msg_type](msg)
            else:
                self.client[each_receiver].msg_handlers[msg.msg_type](msg)
