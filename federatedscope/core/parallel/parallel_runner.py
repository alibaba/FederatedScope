import logging
import time
import os
import copy
import heapq

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from federatedscope.core.fed_runner import StandaloneRunner
from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.core.auxiliaries.feat_engr_builder import \
    get_feat_engr_wrapper
from federatedscope.core.auxiliaries.data_builder import get_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def recv_mode_para(model_para, src_rank):
    for v in model_para.values():
        dist.recv(tensor=v, src=src_rank)


def setup_multigpu_runner(cfg, server_class, client_class, unseen_clients_id,
                          server_resource_info, client_resource_info):
    processes = []
    mp.set_start_method("spawn")

    # init parameter
    client2server_queue = mp.Queue()
    server2client_queues = [
        mp.Queue() for _ in range(1, cfg.federate.process_num)
    ]
    id2comm = dict()
    clients_id_list = []
    client_num_per_process = \
        cfg.federate.client_num // (cfg.federate.process_num - 1)
    for process_id in range(1, cfg.federate.process_num):
        client_ids_start = (process_id - 1) * client_num_per_process + 1
        client_ids_end = client_ids_start + client_num_per_process \
            if process_id != cfg.federate.process_num - 1 \
            else cfg.federate.client_num + 1
        clients_id_list.append(range(client_ids_start, client_ids_end))
        for client_id in range(client_ids_start, client_ids_end):
            id2comm[client_id] = process_id - 1

    # setup server process
    server_rank = 0
    server_process = mp.Process(
        target=run,
        args=(server_rank, cfg.federate.process_num, cfg.federate.master_addr,
              cfg.federate.master_port,
              ServerRunner(rank=server_rank,
                           config=cfg,
                           server_class=server_class,
                           receive_channel=client2server_queue,
                           send_channels=server2client_queues,
                           id2comm=id2comm,
                           unseen_clients_id=unseen_clients_id,
                           resource_info=server_resource_info,
                           client_resource_info=client_resource_info)))
    server_process.start()
    processes.append(server_process)

    # setup client process
    for rank in range(1, cfg.federate.process_num):
        client_runner = ClientRunner(
            rank=rank,
            client_ids=clients_id_list[rank - 1],
            config=cfg,
            client_class=client_class,
            unseen_clients_id=unseen_clients_id,
            receive_channel=server2client_queues[rank - 1],
            send_channel=client2server_queue,
            client_resource_info=client_resource_info)
        p = mp.Process(target=run,
                       args=(rank, cfg.federate.process_num,
                             cfg.federate.master_addr,
                             cfg.federate.master_port, client_runner))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def run(rank, world_size, master_addr, master_port, runner):
    logger.info("Process {} start to run".format(rank))
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    # server process
    runner.setup()
    runner.run()


class StandaloneMultiGPURunner(StandaloneRunner):
    def _set_up(self):
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
            'shared_comm_queue': self.server2client_comm_queue,
            'id2comm': self.id2comm,
            'resource_info': resource_info,
            'client_resource_info': client_resource_info
        }
        return server_data, model, kw

    def _get_client_args(self, client_id=-1, resource_info=None):
        client_data = self.data[client_id]
        kw = {
            'shared_comm_queue': self.client2server_comm_queue,
            'resource_info': resource_info
        }
        return client_data, kw

    def run(self):
        logger.info("Multi-GPU are starting for parallel training ...")
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
        setup_multigpu_runner(self.cfg, self.server_class, self.client_class,
                              self.unseen_clients_id, server_resource_info,
                              client_resource_info)


class Runner(object):
    def __init__(self, rank):
        self.rank = rank
        self.device = torch.device("cuda:{}".format(rank))

    def setup(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


class ServerRunner(Runner):
    def __init__(self, rank, config, server_class, receive_channel,
                 send_channels, id2comm, unseen_clients_id, resource_info,
                 client_resource_info):
        super().__init__(rank)
        self.config = config
        self.server_class = server_class
        self.receive_channel = receive_channel
        self.send_channel = send_channels
        self.id2comm = id2comm
        self.unseen_clients_id = unseen_clients_id
        self.server_id = 0
        self.resource_info = resource_info
        self.client_resource_info = client_resource_info
        self.serial_num_for_msg = 0

    def setup(self):
        self.config.defrost()
        data, modified_cfg = get_data(config=self.config, client_cfgs=None)
        self.config.merge_from_other_cfg(modified_cfg)
        self.config.freeze()
        if self.rank in data:
            self.data = data[self.rank] if self.rank in data else data[1]
            model = get_model(self.config,
                              self.data,
                              backend=self.config.backend)
        else:
            self.data = None
            model = get_model(self.config,
                              data[1],
                              backend=self.config.backend)
        kw = {
            'shared_comm_queue': self.send_channel,
            'id2comm': self.id2comm,
            'resource_info': self.resource_info,
            'client_resource_info': self.client_resource_info
        }

        self.server = self.server_class(
            ID=self.server_id,
            config=self.config,
            data=self.data,
            model=model,
            client_num=self.config.federate.client_num,
            totol_round_num=self.config.federate.total_round_num,
            device=self.device,
            unseen_clients_id=self.unseen_clients_id,
            **kw)

        self.server.model.to(self.device)
        self.template_para = copy.deepcopy(self.server.model.state_dict())
        if self.config.nbafl.use:
            from federatedscope.core.trainers.trainer_nbafl import \
                wrap_nbafl_server
            wrap_nbafl_server(self.server)
        logger.info('Server has been set up ... ')
        _, feat_engr_wrapper_server = get_feat_engr_wrapper(self.config)
        self.server = feat_engr_wrapper_server(self.server)

    def run(self):
        logger.info("ServerRunner {} start to run".format(self.rank))
        server_msg_cache = list()
        while True:
            if not self.receive_channel.empty():
                msg = self.receive_channel.get()
                # For the server, move the received message to a
                # cache for reordering the messages according to
                # the timestamps
                msg.serial_num = self.serial_num_for_msg
                self.serial_num_for_msg += 1
                heapq.heappush(server_msg_cache, msg)
            elif len(server_msg_cache) > 0:
                msg = heapq.heappop(server_msg_cache)
                if self.config.asyn.use and self.config.asyn.aggregator \
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
                if self.config.asyn.use and self.config.asyn.aggregator \
                        == 'time_up':
                    self.server.trigger_for_time_up()
                    if self.client2server_comm_queue.empty() and \
                            len(server_msg_cache) == 0:
                        break
                else:
                    if self.server.is_finish:
                        break
                    else:
                        time.sleep(0.01)

    def _handle_msg(self, msg):
        """
        To simulate the message handling process (used only for the
        standalone mode)
        """
        sender, receiver = msg.sender, msg.receiver
        download_bytes, upload_bytes = msg.count_bytes()
        if msg.msg_type == 'model_para':
            sender_rank = self.id2comm[sender] + 1
            tmp_model_para = copy.deepcopy(self.template_para)
            recv_mode_para(tmp_model_para, sender_rank)
            msg.content = (msg.content[0], tmp_model_para)
        if not isinstance(receiver, list):
            receiver = [receiver]
        for each_receiver in receiver:
            if each_receiver == 0:
                self.server.msg_handlers[msg.msg_type](msg)
                self.server._monitor.track_download_bytes(download_bytes)
            else:
                # should not go here
                logger.warning('server received a wrong message')


class ClientRunner(Runner):
    def __init__(self, rank, client_ids, config, client_class,
                 unseen_clients_id, receive_channel, send_channel,
                 client_resource_info):
        super().__init__(rank)
        self.client_ids = client_ids
        self.config = config
        self.client_class = client_class
        self.unseen_clients_id = unseen_clients_id
        self.base_client_id = client_ids[0]
        self.receive_channel = receive_channel
        self.client2server_comm_queue = send_channel
        self.client_group = dict()
        self.client_resource_info = client_resource_info
        self.is_finish = False

    def setup(self):
        self.config.defrost()
        self.data, modified_cfg = get_data(config=self.config,
                                           client_cfgs=None)
        self.config.merge_from_other_cfg(modified_cfg)
        self.config.freeze()
        self.shared_model = get_model(
            self.config,
            self.data[self.base_client_id],
            backend=self.config.backend
        ) if self.config.federate.share_local_model else None

        server_id = 0

        for client_id in self.client_ids:
            client_data = self.data[client_id]
            kw = {
                'shared_comm_queue': self.client2server_comm_queue,
                'resource_info': self.client_resource_info[client_id]
                if self.client_resource_info is not None else None
            }
            client_specific_config = self.config.clone()
            if self.client_resource_info is not None:
                client_specific_config.defrost()
                client_specific_config.merge_from_other_cfg(
                    self.client_resource_info.get(
                        'client_{}'.format(client_id)))
                client_specific_config.freeze()
            client = self.client_class(
                ID=client_id,
                server_id=server_id,
                config=client_specific_config,
                data=client_data,
                model=self.shared_model
                or get_model(client_specific_config,
                             client_data,
                             backend=self.config.backend),
                device=self.device,
                is_unseen_client=client_id in self.unseen_clients_id,
                **kw)
            client.model.to(self.device)
            logger.info(f'Client {client_id} has been set up ... ')
            self.client_group[client_id] = client
        self.template_para = copy.deepcopy(
            self.client_group[self.base_client_id].model.state_dict())

    def run(self):
        logger.info("ClientRunner {} start to run".format(self.rank))
        for _, client in self.client_group.items():
            client.join_in()
        while True:
            if not self.receive_channel.empty():
                msg = self.receive_channel.get()
                self._handle_msg(msg)
            elif self.is_finish:
                break

    def _handle_msg(self, msg):
        _, receiver = msg.sender, msg.receiver
        msg_type = msg.msg_type
        if msg_type == 'model_para' or msg_type == 'evaluate':
            # recv from server
            recv_mode_para(self.template_para, 0)
            msg.content = self.template_para
        download_bytes, upload_bytes = msg.count_bytes()
        if not isinstance(receiver, list):
            receiver = [receiver]
        for each_receiver in receiver:
            if each_receiver in self.client_ids:
                self.client_group[each_receiver].msg_handlers[msg.msg_type](
                    msg)
                self.client_group[each_receiver]._monitor.track_download_bytes(
                    download_bytes)
        if msg.msg_type == 'finish':
            self.is_finish = True
