import abc
import logging

from collections import deque
import heapq

import numpy as np

from federatedscope.core.workers import Server, Client
from federatedscope.core.gpu_manager import GPUManager
from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.core.auxiliaries.utils import get_resource_info, \
    get_ds_rank
from federatedscope.core.auxiliaries.feat_engr_builder import \
    get_feat_engr_wrapper

logger = logging.getLogger(__name__)


class BaseRunner(object):
    """
    This class is a base class to construct an FL course, which includes \
    ``_set_up()`` and ``run()``.

    Args:
        data: The data used in the FL courses, which are formatted as \
        ``{'ID':data}`` for standalone mode. More details can be found in \
        federatedscope.core.auxiliaries.data_builder .
        server_class: The server class is used for instantiating a ( \
        customized) server.
        client_class: The client class is used for instantiating a ( \
        customized) client.
        config: The configurations of the FL course.
        client_configs: The clients' configurations.

    Attributes:
        data: The data used in the FL courses, which are formatted as \
        ``{'ID':data}`` for standalone mode. More details can be found in \
        federatedscope.core.auxiliaries.data_builder .
        server: The instantiated server.
        client: The instantiate client(s).
        cfg : The configurations of the FL course.
        client_cfgs: The clients' configurations.
        mode: The run mode for FL, ``distributed`` or ``standalone``
        gpu_manager: manager of GPU resource
        resource_info: information of resource
    """
    def __init__(self,
                 data,
                 server_class=Server,
                 client_class=Client,
                 config=None,
                 client_configs=None):
        self.data = data
        self.server_class = server_class
        self.client_class = client_class
        assert config is not None, \
            "When using Runner, you should specify the `config` para"
        if not config.is_ready_for_run:
            config.ready_for_run()
        self.cfg = config
        self.client_cfgs = client_configs
        self.serial_num_for_msg = 0

        self.mode = self.cfg.federate.mode.lower()
        self.gpu_manager = GPUManager(gpu_available=self.cfg.use_gpu,
                                      specified_device=self.cfg.device)

        self.unseen_clients_id = []
        self.feat_engr_wrapper_client, self.feat_engr_wrapper_server = \
            get_feat_engr_wrapper(config)
        if self.cfg.federate.unseen_clients_rate > 0:
            self.unseen_clients_id = np.random.choice(
                np.arange(1, self.cfg.federate.client_num + 1),
                size=max(
                    1,
                    int(self.cfg.federate.unseen_clients_rate *
                        self.cfg.federate.client_num)),
                replace=False).tolist()
        # get resource information
        self.resource_info = get_resource_info(
            config.federate.resource_info_file)

        # Check the completeness of msg_handler.
        self.check()

        # Set up for Runner
        self._set_up()

    @abc.abstractmethod
    def _set_up(self):
        """
        Set up and instantiate the client/server.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_server_args(self, resource_info, client_resource_info):
        """
        Get the args for instantiating the server.

        Args:
            resource_info: information of resource
            client_resource_info: information of client's resource

        Returns:
            (server_data, model, kw): None or data which server holds; model \
            to be aggregated; kwargs dict to instantiate the server.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_client_args(self, client_id, resource_info):
        """
        Get the args for instantiating the server.

        Args:
            client_id: ID of client
            resource_info: information of resource

        Returns:
            (client_data, kw): data which client holds; kwargs dict to \
            instantiate the client.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run(self):
        """
        Launch the FL course

        Returns:
            dict: best results during the FL course
        """
        raise NotImplementedError

    @property
    def ds_rank(self):
        return get_ds_rank()

    def _setup_server(self, resource_info=None, client_resource_info=None):
        """
        Set up and instantiate the server.

        Args:
            resource_info: information of resource
            client_resource_info: information of client's resource

        Returns:
            Instantiate server.
        """
        assert self.server_class is not None, \
            "`server_class` cannot be None."
        self.server_id = 0
        server_data, model, kw = self._get_server_args(resource_info,
                                                       client_resource_info)
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
        if self.cfg.vertical.use:
            from federatedscope.vertical_fl.utils import wrap_vertical_server
            server = wrap_vertical_server(server, self.cfg)
        if self.cfg.fedswa.use:
            from federatedscope.core.workers.wrapper import wrap_swa_server
            server = wrap_swa_server(server)
        logger.info('Server has been set up ... ')
        return self.feat_engr_wrapper_server(server)

    def _setup_client(self,
                      client_id=-1,
                      client_model=None,
                      resource_info=None):
        """
        Set up and instantiate the client.

        Args:
            client_id: ID of client
            client_model: model of client
            resource_info: information of resource

        Returns:
            Instantiate client.
        """
        assert self.client_class is not None, \
            "`client_class` cannot be None"
        self.server_id = 0
        client_data, kw = self._get_client_args(client_id, resource_info)
        client_specific_config = self.cfg.clone()
        if self.client_cfgs:
            client_specific_config.defrost()
            client_specific_config.merge_from_other_cfg(
                self.client_cfgs.get('client_{}'.format(client_id)))
            client_specific_config.freeze()
        client_device = self._server_device if \
            self.cfg.federate.share_local_model else \
            self.gpu_manager.auto_choice()
        client = self.client_class(
            ID=client_id,
            server_id=self.server_id,
            config=client_specific_config,
            data=client_data,
            model=client_model or get_model(
                client_specific_config, client_data, backend=self.cfg.backend),
            device=client_device,
            is_unseen_client=client_id in self.unseen_clients_id,
            **kw)

        if self.cfg.vertical.use:
            from federatedscope.vertical_fl.utils import wrap_vertical_client
            client = wrap_vertical_client(client, config=self.cfg)

        if client_id == -1:
            logger.info('Client (address {}:{}) has been set up ... '.format(
                self.client_address['host'], self.client_address['port']))
        else:
            logger.info(f'Client {client_id} has been set up ... ')

        return self.feat_engr_wrapper_client(client)

    def check(self):
        """
        Check the completeness of Server and Client.

        """
        if not self.cfg.check_completeness:
            return
        try:
            import os
            import networkx as nx
            import matplotlib.pyplot as plt
            # Build check graph
            G = nx.DiGraph()
            flags = {0: 'Client', 1: 'Server'}
            msg_handler_dicts = [
                self.client_class.get_msg_handler_dict(),
                self.server_class.get_msg_handler_dict()
            ]
            for flag, msg_handler_dict in zip(flags.keys(), msg_handler_dicts):
                role, oppo = flags[flag], flags[(flag + 1) % 2]
                for msg_in, (handler, msgs_out) in \
                        msg_handler_dict.items():
                    for msg_out in msgs_out:
                        msg_in_key = f'{oppo}_{msg_in}'
                        handler_key = f'{role}_{handler}'
                        msg_out_key = f'{role}_{msg_out}'
                        G.add_node(msg_in_key, subset=1)
                        G.add_node(handler_key, subset=0 if flag else 2)
                        G.add_node(msg_out_key, subset=1)
                        G.add_edge(msg_in_key, handler_key)
                        G.add_edge(handler_key, msg_out_key)
            pos = nx.multipartite_layout(G)
            plt.figure(figsize=(20, 15))
            nx.draw(G,
                    pos,
                    with_labels=True,
                    node_color='white',
                    node_size=800,
                    width=1.0,
                    arrowsize=25,
                    arrowstyle='->')
            fig_path = os.path.join(self.cfg.outdir, 'msg_handler.png')
            plt.savefig(fig_path)
            if nx.has_path(G, 'Client_join_in', 'Server_finish'):
                if nx.is_weakly_connected(G):
                    logger.info(f'Completeness check passes! Save check '
                                f'results in {fig_path}.')
                else:
                    logger.warning(f'Completeness check raises warning for '
                                   f'some handlers not in FL process! Save '
                                   f'check results in {fig_path}.')
            else:
                logger.error(f'Completeness check fails for there is no'
                             f'path from `join_in` to `finish`! Save '
                             f'check results in {fig_path}.')
        except Exception as error:
            logger.warning(f'Completeness check failed for {error}!')
        return


class StandaloneRunner(BaseRunner):
    def _set_up(self):
        """
        To set up server and client for standalone mode.
        """
        self.is_run_online = True if self.cfg.federate.online_aggr else False
        self.shared_comm_queue = deque()

        if self.cfg.backend == 'torch':
            import torch
            torch.set_num_threads(1)

        assert self.cfg.federate.client_num != 0, \
            "In standalone mode, self.cfg.federate.client_num should be " \
            "non-zero. " \
            "This is usually cased by using synthetic data and users not " \
            "specify a non-zero value for client_num"

        if self.cfg.federate.method == "global":
            self.cfg.defrost()
            self.cfg.federate.client_num = 1
            self.cfg.federate.sample_client_num = 1
            self.cfg.freeze()

        # sample resource information
        if self.resource_info is not None:
            if len(self.resource_info) < self.cfg.federate.client_num + 1:
                replace = True
                logger.warning(
                    f"Because the provided the number of resource information "
                    f"{len(self.resource_info)} is less than the number of "
                    f"participants {self.cfg.federate.client_num + 1}, one "
                    f"candidate might be selected multiple times.")
            else:
                replace = False
            sampled_index = np.random.choice(
                list(self.resource_info.keys()),
                size=self.cfg.federate.client_num + 1,
                replace=replace)
            server_resource_info = self.resource_info[sampled_index[0]]
            client_resource_info = [
                self.resource_info[x] for x in sampled_index[1:]
            ]
        else:
            server_resource_info = None
            client_resource_info = None

        self.server = self._setup_server(
            resource_info=server_resource_info,
            client_resource_info=client_resource_info)

        self.client = dict()
        # assume the client-wise data are consistent in their input&output
        # shape
        if self.cfg.federate.online_aggr:
            self._shared_client_model = get_model(
                self.cfg, self.data[1], backend=self.cfg.backend
            ) if self.cfg.federate.share_local_model else None
        else:
            self._shared_client_model = self.server.model \
                if self.cfg.federate.share_local_model else None
        for client_id in range(1, self.cfg.federate.client_num + 1):
            self.client[client_id] = self._setup_client(
                client_id=client_id,
                client_model=self._shared_client_model,
                resource_info=client_resource_info[client_id - 1]
                if client_resource_info is not None else None)

        # in standalone mode, by default, we print the trainer info only
        # once for better logs readability
        trainer_representative = self.client[1].trainer
        if trainer_representative is not None and hasattr(
                trainer_representative, 'print_trainer_meta_info'):
            trainer_representative.print_trainer_meta_info()

    def _get_server_args(self, resource_info=None, client_resource_info=None):
        if self.server_id in self.data:
            server_data = self.data[self.server_id]
            model = get_model(self.cfg, server_data, backend=self.cfg.backend)
        else:
            server_data = None
            data_representative = self.data[1]
            model = get_model(
                self.cfg, data_representative, backend=self.cfg.backend
            )  # get the model according to client's data if the server
            # does not own data
        kw = {
            'shared_comm_queue': self.shared_comm_queue,
            'resource_info': resource_info,
            'client_resource_info': client_resource_info
        }
        return server_data, model, kw

    def _get_client_args(self, client_id=-1, resource_info=None):
        client_data = self.data[client_id]
        kw = {
            'shared_comm_queue': self.shared_comm_queue,
            'resource_info': resource_info
        }
        return client_data, kw

    def run(self):
        for each_client in self.client:
            # Launch each client
            self.client[each_client].join_in()

        if self.is_run_online:
            self._run_simulation_online()
        else:
            self._run_simulation()
        # TODO: avoid using private attr
        self.server._monitor.finish_fed_runner(fl_mode=self.mode)
        return self.server.best_results

    def _handle_msg(self, msg, rcv=-1):
        """
        To simulate the message handling process (used only for the \
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

    def _run_simulation_online(self):
        """
        Run for online aggregation.
        Any broadcast operation would be executed client-by-clien to avoid \
        the existence of #clients messages at the same time. Currently, \
        only consider centralized topology \
        """
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

    def _run_simulation(self):
        """
        Run for standalone simulation (W/O online aggr)
        """
        server_msg_cache = list()
        while True:
            if len(self.shared_comm_queue) > 0:
                msg = self.shared_comm_queue.popleft()
                if not self.cfg.vertical.use and msg.receiver == [
                        self.server_id
                ]:
                    # For the server, move the received message to a
                    # cache for reordering the messages according to
                    # the timestamps
                    msg.serial_num = self.serial_num_for_msg
                    self.serial_num_for_msg += 1
                    heapq.heappush(server_msg_cache, msg)
                else:
                    self._handle_msg(msg)
            elif len(server_msg_cache) > 0:
                msg = heapq.heappop(server_msg_cache)
                if self.cfg.asyn.use and self.cfg.asyn.aggregator \
                        == 'time_up':
                    # When the timestamp of the received message beyond
                    # the deadline for the currency round, trigger the
                    # time up event first and push the message back to
                    # the cache
                    if self.server.trigger_for_time_up(msg.timestamp):
                        heapq.heappush(server_msg_cache, msg)
                    else:
                        self._handle_msg(msg)
                else:
                    self._handle_msg(msg)
            else:
                if self.cfg.asyn.use and self.cfg.asyn.aggregator \
                        == 'time_up':
                    self.server.trigger_for_time_up()
                    if len(self.shared_comm_queue) == 0 and \
                            len(server_msg_cache) == 0:
                        break
                else:
                    # terminate when shared_comm_queue and
                    # server_msg_cache are all empty
                    break


class DistributedRunner(BaseRunner):
    def _set_up(self):
        """
        To set up server or client for distributed mode.
        """
        # sample resource information
        if self.resource_info is not None:
            sampled_index = np.random.choice(list(self.resource_info.keys()))
            sampled_resource = self.resource_info[sampled_index]
        else:
            sampled_resource = None

        self.server_address = {
            'host': self.cfg.distribute.server_host,
            'port': self.cfg.distribute.server_port + self.ds_rank
        }
        if self.cfg.distribute.role == 'server':
            self.server = self._setup_server(resource_info=sampled_resource)
        elif self.cfg.distribute.role == 'client':
            # When we set up the client in the distributed mode, we assume
            # the server has been set up and number with #0
            self.client_address = {
                'host': self.cfg.distribute.client_host,
                'port': self.cfg.distribute.client_port + self.ds_rank
            }
            self.client = self._setup_client(resource_info=sampled_resource)

    def _get_server_args(self, resource_info, client_resource_info):
        server_data = self.data
        model = get_model(self.cfg, server_data, backend=self.cfg.backend)
        kw = self.server_address
        kw.update({'resource_info': resource_info})
        return server_data, model, kw

    def _get_client_args(self, client_id, resource_info):
        client_data = self.data
        kw = self.client_address
        kw['server_host'] = self.server_address['host']
        kw['server_port'] = self.server_address['port']
        kw['resource_info'] = resource_info
        return client_data, kw

    def run(self):
        if self.cfg.distribute.role == 'server':
            self.server.run()
            return self.server.best_results
        elif self.cfg.distribute.role == 'client':
            self.client.join_in()
            self.client.run()


# TODO: remove FedRunner (keep now for forward compatibility)
class FedRunner(object):
    """
    This class is used to construct an FL course, which includes `_set_up`
    and `run`.

    Arguments:
        data: The data used in the FL courses, which are formatted as \
        ``{'ID':data}`` for standalone mode. More details can be found in \
        federatedscope.core.auxiliaries.data_builder .
        server_class: The server class is used for instantiating a ( \
        customized) server.
        client_class: The client class is used for instantiating a ( \
        customized) client.
        config: The configurations of the FL course.
        client_configs: The clients' configurations.

    Warnings:
        ``FedRunner`` will be removed in the future, consider \
        using ``StandaloneRunner`` or ``DistributedRunner`` instead!
    """
    def __init__(self,
                 data,
                 server_class=Server,
                 client_class=Client,
                 config=None,
                 client_configs=None):
        logger.warning('`federate.core.fed_runner.FedRunner` will be '
                       'removed in the future, please use'
                       '`federate.core.fed_runner.get_runner` to get '
                       'Runner.')
        self.data = data
        self.server_class = server_class
        self.client_class = client_class
        assert config is not None, \
            "When using FedRunner, you should specify the `config` para"
        if not config.is_ready_for_run:
            config.ready_for_run()
        self.cfg = config
        self.client_cfgs = client_configs

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
        # get resource information
        self.resource_info = get_resource_info(
            config.federate.resource_info_file)

        # Check the completeness of msg_handler.
        self.check()

    def setup(self):
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
            self.cfg.defrost()
            self.cfg.federate.client_num = 1
            self.cfg.federate.sample_client_num = 1
            self.cfg.freeze()

        # sample resource information
        if self.resource_info is not None:
            if len(self.resource_info) < self.cfg.federate.client_num + 1:
                replace = True
                logger.warning(
                    f"Because the provided the number of resource information "
                    f"{len(self.resource_info)} is less than the number of "
                    f"participants {self.cfg.federate.client_num+1}, one "
                    f"candidate might be selected multiple times.")
            else:
                replace = False
            sampled_index = np.random.choice(
                list(self.resource_info.keys()),
                size=self.cfg.federate.client_num + 1,
                replace=replace)
            server_resource_info = self.resource_info[sampled_index[0]]
            client_resource_info = [
                self.resource_info[x] for x in sampled_index[1:]
            ]
        else:
            server_resource_info = None
            client_resource_info = None

        self.server = self._setup_server(
            resource_info=server_resource_info,
            client_resource_info=client_resource_info)

        self.client = dict()

        # assume the client-wise data are consistent in their input&output
        # shape
        self._shared_client_model = get_model(
            self.cfg, self.data[1], backend=self.cfg.backend
        ) if self.cfg.federate.share_local_model else None

        for client_id in range(1, self.cfg.federate.client_num + 1):
            self.client[client_id] = self._setup_client(
                client_id=client_id,
                client_model=self._shared_client_model,
                resource_info=client_resource_info[client_id - 1]
                if client_resource_info is not None else None)

    def _setup_for_distributed(self):
        """
        To set up server or client for distributed mode.
        """

        # sample resource information
        if self.resource_info is not None:
            sampled_index = np.random.choice(list(self.resource_info.keys()))
            sampled_resource = self.resource_info[sampled_index]
        else:
            sampled_resource = None

        self.server_address = {
            'host': self.cfg.distribute.server_host,
            'port': self.cfg.distribute.server_port
        }
        if self.cfg.distribute.role == 'server':
            self.server = self._setup_server(resource_info=sampled_resource)
        elif self.cfg.distribute.role == 'client':
            # When we set up the client in the distributed mode, we assume
            # the server has been set up and number with #0
            self.client_address = {
                'host': self.cfg.distribute.client_host,
                'port': self.cfg.distribute.client_port
            }
            self.client = self._setup_client(resource_info=sampled_resource)

    def run(self):
        """
        To run an FL course, which is called after server/client has been
        set up.
        For the standalone mode, a shared message queue will be set up to
        simulate ``receiving message``.
        """
        self.setup()
        if self.mode == 'standalone':
            # trigger the FL course
            for each_client in self.client:
                self.client[each_client].join_in()

            if self.cfg.federate.online_aggr:
                # any broadcast operation would be executed client-by-client
                # to avoid the existence of #clients messages at the same time.
                # currently, only consider centralized topology
                self._run_simulation_online()

            else:
                self._run_simulation()

            self.server._monitor.finish_fed_runner(fl_mode=self.mode)

            return self.server.best_results

        elif self.mode == 'distributed':
            if self.cfg.distribute.role == 'server':
                self.server.run()
                return self.server.best_results
            elif self.cfg.distribute.role == 'client':
                self.client.join_in()
                self.client.run()

    def _run_simulation_online(self):
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

    def _run_simulation(self):
        server_msg_cache = list()
        while True:
            if len(self.shared_comm_queue) > 0:
                msg = self.shared_comm_queue.popleft()
                if msg.receiver == [self.server_id]:
                    # For the server, move the received message to a
                    # cache for reordering the messages according to
                    # the timestamps
                    heapq.heappush(server_msg_cache, msg)
                else:
                    self._handle_msg(msg)
            elif len(server_msg_cache) > 0:
                msg = heapq.heappop(server_msg_cache)
                if self.cfg.asyn.use and self.cfg.asyn.aggregator \
                        == 'time_up':
                    # When the timestamp of the received message beyond
                    # the deadline for the currency round, trigger the
                    # time up event first and push the message back to
                    # the cache
                    if self.server.trigger_for_time_up(msg.timestamp):
                        heapq.heappush(server_msg_cache, msg)
                    else:
                        self._handle_msg(msg)
                else:
                    self._handle_msg(msg)
            else:
                if self.cfg.asyn.use and self.cfg.asyn.aggregator \
                        == 'time_up':
                    self.server.trigger_for_time_up()
                    if len(self.shared_comm_queue) == 0 and \
                            len(server_msg_cache) == 0:
                        break
                else:
                    # terminate when shared_comm_queue and
                    # server_msg_cache are all empty
                    break

    def _setup_server(self, resource_info=None, client_resource_info=None):
        """
        Set up the server
        """
        self.server_id = 0
        if self.mode == 'standalone':
            if self.server_id in self.data:
                server_data = self.data[self.server_id]
                model = get_model(self.cfg,
                                  server_data,
                                  backend=self.cfg.backend)
            else:
                server_data = None
                data_representative = self.data[1]
                model = get_model(
                    self.cfg, data_representative, backend=self.cfg.backend
                )  # get the model according to client's data if the server
                # does not own data
            kw = {
                'shared_comm_queue': self.shared_comm_queue,
                'resource_info': resource_info,
                'client_resource_info': client_resource_info
            }
        elif self.mode == 'distributed':
            server_data = self.data
            model = get_model(self.cfg, server_data, backend=self.cfg.backend)
            kw = self.server_address
            kw.update({'resource_info': resource_info})
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

        logger.info('Server has been set up ... ')

        return server

    def _setup_client(self,
                      client_id=-1,
                      client_model=None,
                      resource_info=None):
        """
        Set up the client
        """
        self.server_id = 0
        if self.mode == 'standalone':
            client_data = self.data[client_id]
            kw = {
                'shared_comm_queue': self.shared_comm_queue,
                'resource_info': resource_info
            }
        elif self.mode == 'distributed':
            client_data = self.data
            kw = self.client_address
            kw['server_host'] = self.server_address['host']
            kw['server_port'] = self.server_address['port']
            kw['resource_info'] = resource_info
        else:
            raise ValueError('Mode {} is not provided'.format(
                self.cfg.mode.type))

        if self.client_class:
            client_specific_config = self.cfg.clone()
            if self.client_cfgs and \
                    self.client_cfgs.get('client_{}'.format(client_id)):
                client_specific_config.defrost()
                client_specific_config.merge_from_other_cfg(
                    self.client_cfgs.get('client_{}'.format(client_id)))
                client_specific_config.freeze()
            client_device = self._server_device if \
                self.cfg.federate.share_local_model else \
                self.gpu_manager.auto_choice()
            client = self.client_class(ID=client_id,
                                       server_id=self.server_id,
                                       config=client_specific_config,
                                       data=client_data,
                                       model=client_model
                                       or get_model(client_specific_config,
                                                    client_data,
                                                    backend=self.cfg.backend),
                                       device=client_device,
                                       is_unseen_client=client_id
                                       in self.unseen_clients_id,
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

    def check(self):
        """
        Check the completeness of Server and Client.

        """
        if not self.cfg.check_completeness:
            return
        try:
            import os
            import networkx as nx
            import matplotlib.pyplot as plt
            # Build check graph
            G = nx.DiGraph()
            flags = {0: 'Client', 1: 'Server'}
            msg_handler_dicts = [
                self.client_class.get_msg_handler_dict(),
                self.server_class.get_msg_handler_dict()
            ]
            for flag, msg_handler_dict in zip(flags.keys(), msg_handler_dicts):
                role, oppo = flags[flag], flags[(flag + 1) % 2]
                for msg_in, (handler, msgs_out) in \
                        msg_handler_dict.items():
                    for msg_out in msgs_out:
                        msg_in_key = f'{oppo}_{msg_in}'
                        handler_key = f'{role}_{handler}'
                        msg_out_key = f'{role}_{msg_out}'
                        G.add_node(msg_in_key, subset=1)
                        G.add_node(handler_key, subset=0 if flag else 2)
                        G.add_node(msg_out_key, subset=1)
                        G.add_edge(msg_in_key, handler_key)
                        G.add_edge(handler_key, msg_out_key)
            pos = nx.multipartite_layout(G)
            plt.figure(figsize=(20, 15))
            nx.draw(G,
                    pos,
                    with_labels=True,
                    node_color='white',
                    node_size=800,
                    width=1.0,
                    arrowsize=25,
                    arrowstyle='->')
            fig_path = os.path.join(self.cfg.outdir, 'msg_handler.png')
            plt.savefig(fig_path)
            if nx.has_path(G, 'Client_join_in', 'Server_finish'):
                if nx.is_weakly_connected(G):
                    logger.info(f'Completeness check passes! Save check '
                                f'results in {fig_path}.')
                else:
                    logger.warning(f'Completeness check raises warning for '
                                   f'some handlers not in FL process! Save '
                                   f'check results in {fig_path}.')
            else:
                logger.error(f'Completeness check fails for there is no'
                             f'path from `join_in` to `finish`! Save '
                             f'check results in {fig_path}.')
        except Exception as error:
            logger.warning(f'Completeness check failed for {error}!')
        return
