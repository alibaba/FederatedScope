import abc
import logging

from collections import deque
import heapq
import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)
import time

import numpy as np

from federatedscope.core.workers import Server, Client
from federatedscope.core.gpu_manager import GPUManager
from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.core.auxiliaries.data_builder import merge_data
from federatedscope.core.auxiliaries.utils import get_resource_info

logger = logging.getLogger(__name__)


def get_runner(data, server_class, client_class, config, client_configs=None):
    # Instantiate a Runner based on a configuration file
    mode = config.federate.mode.lower()
    runner_dict = {
        'standalone': StandaloneRunner
        if not config.federate.parallel else StandaloneParallelRunner,
        'distributed': DistributedRunner,
    }
    return runner_dict[mode](data=data,
                             server_class=server_class,
                             client_class=client_class,
                             config=config,
                             client_configs=client_configs)


class BaseRunner(object):
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
        client_configs: The clients' configurations.
    """
    def __init__(self,
                 data,
                 server_class=Server,
                 client_class=Client,
                 config=None,
                 client_configs=None):
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        self.data = data
        self.server_class = server_class
        self.client_class = client_class
        assert config is not None, \
            "When using Runner, you should specify the `config` para"
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

        # Set up for Runner
        self._set_up()

    @abc.abstractmethod
    def _set_up(self):
        """
        Set up client and/or server
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
            server_data: None or data which server holds.
            model: model to be aggregated.
            kw: kwargs dict to instantiate the server.
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
            client_data: data which client holds.
            kw: kwargs dict to instantiate the client.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run(self):
        """
        Launch the worker

        Returns:
            best_results: best results during the FL course
        """
        raise NotImplementedError

    def _setup_server(self, resource_info=None, client_resource_info=None):
        """
        Set up the server
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
        logger.info('Server has been set up ... ')
        return server

    def _setup_client(self,
                      client_id=-1,
                      client_model=None,
                      resource_info=None):
        """
        Set up the Client
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
        client = self.client_class(ID=client_id,
                                   server_id=self.server_id,
                                   config=client_specific_config,
                                   data=client_data,
                                   model=client_model
                                   or get_model(client_specific_config.model,
                                                client_data,
                                                backend=self.cfg.backend),
                                   device=client_device,
                                   is_unseen_client=client_id
                                   in self.unseen_clients_id,
                                   **kw)

        if client_id == -1:
            logger.info('Client (address {}:{}) has been set up ... '.format(
                self.client_address['host'], self.client_address['port']))
        else:
            logger.info(f'Client {client_id} has been set up ... ')

        return client


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
        self._shared_client_model = get_model(
            self.cfg.model, self.data[1], backend=self.cfg.backend
        ) if self.cfg.federate.share_local_model else None
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
            model = get_model(self.cfg.model,
                              server_data,
                              backend=self.cfg.backend)
        else:
            server_data = None
            data_representative = self.data[1]
            model = get_model(
                self.cfg.model, data_representative, backend=self.cfg.backend
            )  # get the model according to client's data if the server
            # does not own data
        kw = {
            'comm_queue': self.shared_comm_queue,
            'resource_info': resource_info,
            'client_resource_info': client_resource_info
        }
        return server_data, model, kw

    def _get_client_args(self, client_id=-1, resource_info=None):
        client_data = self.data[client_id]
        kw = {
            'comm_queue': self.shared_comm_queue,
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

    def _run_simulation_online(self):
        """
        Run for online aggregation.
        Any broadcast operation would be executed client-by-clien to avoid
        the existence of #clients messages at the same time. Currently,
        only consider centralized topology
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


class StandaloneParallelRunner(StandaloneRunner):
    def _set_up(self):
        """
        To set up server and client for parallel training in standalone mode.
        """

        if self.cfg.backend == 'torch':
            import torch
            torch.set_num_threads(1)

        assert self.cfg.federate.client_num != 0, \
            "In standalone mode, self.cfg.federate.client_num should be " \
            "non-zero. " \
            "This is usually cased by using synthetic data and users not " \
            "specify a non-zero value for client_num"

        if self.cfg.federate.client_num < self.cfg.federate.process_num:
            logger.warning('The process number is more than client number')
            self.cfg.federate.process_num = self.cfg.federate.client_num

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

        self.manager = multiprocessing.Manager()
        self.client2server_comm_queue = self.manager.Queue()
        self.server2client_comm_queue = list()
        for process_id in range(self.cfg.federate.process_num):
            self.server2client_comm_queue.append(self.manager.Queue())

        # Instantiate ClientRunner for parallel training
        self.client_runners = list()
        self.id2comm = dict()
        client_num_per_process = \
            self.cfg.federate.client_num // self.cfg.federate.process_num
        for process_id in range(self.cfg.federate.process_num):
            client_ids_start = process_id * client_num_per_process + 1
            client_ids_end = client_ids_start + client_num_per_process \
                if process_id != self.cfg.federate.process_num - 1 \
                else self.cfg.federate.client_num + 1
            runner_device = f'cuda:{process_id%8}'
            client_runner = ClientRunner(
                client_ids=range(client_ids_start, client_ids_end),
                device=runner_device,
                config=self.cfg,
                data={
                    k: v
                    for k, v in self.data.items()
                    if k in range(client_ids_start, client_ids_end)
                },
                client_class=self.client_class,
                unseen_clients_id=self.unseen_clients_id,
                receive_channel=self.server2client_comm_queue[process_id],
                send_channel=self.client2server_comm_queue,
                # TODO
                client_resource_info=client_resource_info)
            self.client_runners.append(client_runner)
            for client_id in range(client_ids_start, client_ids_end):
                self.id2comm[client_id] = process_id

        self.server = self._setup_server(
            resource_info=server_resource_info,
            client_resource_info=client_resource_info)

    def _get_server_args(self, resource_info=None, client_resource_info=None):
        if self.server_id in self.data:
            server_data = self.data[self.server_id]
            model = get_model(self.cfg.model,
                              server_data,
                              backend=self.cfg.backend)
        else:
            server_data = None
            data_representative = self.data[1]
            model = get_model(
                self.cfg.model, data_representative, backend=self.cfg.backend
            )  # get the model according to client's data if the server
            # does not own data
        kw = {
            'comm_queue': self.server2client_comm_queue,
            'id2comm': self.id2comm,
            'resource_info': resource_info,
            'client_resource_info': client_resource_info
        }
        return server_data, model, kw

    def _get_client_args(self, client_id=-1, resource_info=None):
        client_data = self.data[client_id]
        kw = {
            'comm_queue': self.client2server_comm_queue,
            'resource_info': resource_info
        }
        return client_data, kw

    def run(self):
        def _print_error(error):
            logger.error(error)

        logger.info('Multi-processes are starting for parallel training ...')
        self.pool = multiprocessing.Pool(
            processes=self.cfg.federate.process_num)
        for client_runner in self.client_runners:
            self.pool.apply_async(client_runner.run,
                                  error_callback=_print_error)
        self.pool.close()

        # TODO: Can online aggregation work?
        self._run_simulation()

        self.pool.join()
        self.server._monitor.finish_fed_runner(fl_mode=self.mode)

        return self.server.best_results

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

    def _run_simulation(self):
        """
        Run for standalone simulation (W/O online aggr)
        """
        server_msg_cache = list()
        while True:
            if not self.client2server_comm_queue.empty():
                msg = self.client2server_comm_queue.get()
                # For the server, move the received message to a
                # cache for reordering the messages according to
                # the timestamps
                heapq.heappush(server_msg_cache, msg)
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
                    if self.client2server_comm_queue.empty() and \
                            len(server_msg_cache) == 0:
                        break
                    else:
                        # terminate when shared_comm_queue and
                        # server_msg_cache are all empty
                        time.sleep(0.01)
                        # break


class ClientRunner(StandaloneParallelRunner):
    """Simulate a group of clients in standalone mode
    """
    def __init__(self,
                 client_ids,
                 device,
                 config,
                 data,
                 client_class,
                 unseen_clients_id,
                 receive_channel,
                 send_channel,
                 client_resource_info=None,
                 client_cfgs=None):

        self.data = data
        self.client_ids = client_ids
        self.base_client_id = client_ids[0]
        self.receive_channel = receive_channel
        self.client2server_comm_queue = send_channel

        self.client_group = dict()
        self.shared_model = get_model(
            config.model, data[self.base_client_id], backend=config.backend
        ) if config.federate.share_local_model else None
        server_id = 0

        for client_id in client_ids:
            client_data, kw = self._get_client_args(
                client_id, client_resource_info[client_id]
                if client_resource_info is not None else None)
            client_specific_config = config.clone()
            # TODO
            if client_cfgs is not None:
                client_specific_config.defrost()
                client_specific_config.merge_from_other_cfg(
                    client_cfgs.get('client_{}'.format(client_id)))
                client_specific_config.freeze()
            client_device = device

            client = client_class(ID=client_id,
                                  server_id=server_id,
                                  config=client_specific_config,
                                  data=client_data,
                                  model=self.shared_model
                                  or get_model(client_specific_config.model,
                                               client_data,
                                               backend=config.backend),
                                  device=client_device,
                                  is_unseen_client=client_id
                                  in unseen_clients_id,
                                  **kw)

            logger.info(f'Client {client_id} has been set up ... ')
            self.client_group[client_id] = client

    def run(self):
        for id, client in self.client_group.items():
            client.join_in()
        while True:
            if not self.receive_channel.empty():
                msg = self.receive_channel.get()
                self._handle_msg(msg)

    def _handle_msg(self, msg, rcv=-1):
        if rcv != -1:
            # simulate broadcast one-by-one
            self.client_group[rcv].msg_handlers[msg.msg_type](msg)
            return

        _, receiver = msg.sender, msg.receiver
        download_bytes, upload_bytes = msg.count_bytes()
        if not isinstance(receiver, list):
            receiver = [receiver]
        for each_receiver in receiver:
            if each_receiver in self.client_ids:
                self.client_group[each_receiver].msg_handlers[msg.msg_type](
                    msg)
                self.client_group[each_receiver]._monitor.track_download_bytes(
                    download_bytes)


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

    def _get_server_args(self, resource_info, client_resource_info):
        server_data = self.data
        model = get_model(self.cfg.model,
                          server_data,
                          backend=self.cfg.backend)
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
        data: The data used in the FL courses, which are formatted as {
        'ID':data} for standalone mode. More details can be found in
        federatedscope.core.auxiliaries.data_builder .
        server_class: The server class is used for instantiating a (
        customized) server.
        client_class: The client class is used for instantiating a (
        customized) client.
        config: The configurations of the FL course.
        client_configs: The clients' configurations.
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
            self.cfg.model, self.data[1], backend=self.cfg.backend
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
                model = get_model(self.cfg.model,
                                  server_data,
                                  backend=self.cfg.backend)
            else:
                server_data = None
                data_representative = self.data[1]
                model = get_model(
                    self.cfg.model,
                    data_representative,
                    backend=self.cfg.backend
                )  # get the model according to client's data if the server
                # does not own data
            kw = {
                'comm_queue': self.shared_comm_queue,
                'resource_info': resource_info,
                'client_resource_info': client_resource_info
            }
        elif self.mode == 'distributed':
            server_data = self.data
            model = get_model(self.cfg.model,
                              server_data,
                              backend=self.cfg.backend)
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
                'comm_queue': self.shared_comm_queue,
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
