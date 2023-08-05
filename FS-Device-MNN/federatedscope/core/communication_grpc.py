import logging
import grpc
import numpy as np

from federatedscope.core.proto import gRPC_comm_manager_pb2, \
    gRPC_comm_manager_pb2_grpc
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.gRPC_server import ServiceFunc
from multiprocessing import Process, Queue, Manager
from concurrent import futures
from tqdm import tqdm

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 512


def get_file_chunks(path_file):
    with open(path_file, 'rb') as f:
        chunks = f.read()
    return chunks


def pack_file_request(state, chunk, client_id):
    assert isinstance(client_id, list)
    file_info = gRPC_comm_manager_pb2.FileInfo(sender=0, state=state, n_sample=0, client_id=client_id)
    request = gRPC_comm_manager_pb2.FileRequest(
        info=file_info,
        chunk=chunk
    )
    return request


class GrpcRequestListener(Process):
    def __init__(self, save_path, host, port, max_workers, queue, compression_method, model_type):
        super(GrpcRequestListener, self).__init__()
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.server_funcs = ServiceFunc(save_path, queue, model_type)
        self.server = None
        self.compression_method = compression_method

    def run(self):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=self.max_workers),
                                  options=[
                                      ("grpc.max_send_message_length",
                                       global_cfg.distribute.grpc_max_send_message_length),
                                      ("grpc.max_receive_message_length",
                                       global_cfg.distribute.grpc_max_receive_message_length),
                                      ("grpc.enable_http_proxy",
                                       global_cfg.distribute.grpc_enable_http_proxy)],
                                  compression=self.compression_method)
        gRPC_comm_manager_pb2_grpc.add_gRPCComServeFuncServicer_to_server(self.server_funcs, self.server)
        self.server.add_insecure_port(f"{self.host}:{self.port}")
        self.server.start()
        self.server.wait_for_termination()


class GrpcSender(Process):
    def __init__(self, queue, neighbors, compression_method):
        super(GrpcSender, self).__init__()
        self.queue_sender = queue
        self.neighbors = neighbors
        self.compression_method = compression_method

    def run(self):
        while True:
            msg_type, receiver_address, obj = self.queue_sender.get(block=True)
            channel = grpc.insecure_channel(receiver_address,
                                            options=(('grpc.enable_http_proxy', 0),),
                                            compression=self.compression_method)
            stub = gRPC_comm_manager_pb2_grpc.gRPCComServeFuncStub(channel)
            try:
                if msg_type == "model_para":
                    client_idx, state, chunk = obj
                    stub.sendMnnModel4Train(pack_file_request(state, chunk, client_idx))
                elif msg_type == "evaluate":
                    client_idx, state, chunk = obj
                    stub.sendMnnModel4Test(pack_file_request(state, chunk, client_idx))
                else:
                    request = obj.transform(to_list=True)
                    stub.sendMessage(request)
            except grpc._channel._InactiveRpcError:
                logger.info(f"Failed to send message {msg_type} to {receiver_address}")
                remove = False
                for key in self.neighbors.keys():
                    if self.neighbors[key] == receiver_address:
                        del self.neighbors[key]
                        remove = True
                        break
                if remove:
                    logger.info(f"Send {msg_type} message failed, remove {receiver_address} from list, {len(self.neighbors)} clients are remained!")
                else:
                    logger.info(f"Failed to remove {receiver_address} from list!")
                if msg_type == "evaluate":
                    new_address = np.random.choice(list(self.neighbors.values()), 1)
                    self.queue_sender.put((msg_type, new_address, obj))
            channel.close()


def get_compression_method(compression):
    if compression.lower() == "gzip":
        return grpc.Compression.Gzip
    elif compression.lower() == "deflate":
        return grpc.Compression.Deflate
    else:
        return None


class gRPCCommManager(object):
    """
        The implementation of gRPCCommManager is referred to the tutorial on
        https://grpc.io/docs/languages/python/
    """

    def __init__(self, save_path, host='0.0.0.0', port='50050', client_num=2, compression=None, model_type=""):
        self.host = host
        self.port = port
        self.neighbors = Manager().dict()
        self.monitor = None  # used to track the communication related metrics

        compression_method = get_compression_method(compression)

        # Start processes to send messages
        self.queue_send = Queue()
        self.senders = [GrpcSender(self.queue_send, self.neighbors, compression_method) for _ in range(30)]
        logger.info(f"Start process to send messages with compression method {compression} ...")
        for sender in tqdm(self.senders):
            sender.start()

        # Start a process to receive messages from clients
        logger.info(f"Start process to listen {self.host}:{self.port} with compression method {compression}...")
        self.queue_receive = Queue()
        self.listener = GrpcRequestListener(save_path, self.host, self.port, client_num, self.queue_receive, compression_method, model_type)
        self.listener.start()

        self.id_to_assign = 0

    def close(self):
        for sender in self.senders:
            sender.terminate()
            sender.join()

        self.listener.terminate()
        self.listener.join()

    def get_new_id(self):
        self.id_to_assign += 1
        return self.id_to_assign

    def add_neighbors(self, neighbor_id, address):
        if isinstance(address, dict):
            self.neighbors[neighbor_id] = '{}:{}'.format(
                address['host'], address['port'])
        elif isinstance(address, str):
            self.neighbors[neighbor_id] = address
        else:
            raise TypeError(f"The type of address ({type(address)}) is not "
                            "supported yet")

    def get_neighbors(self, neighbor_id=None):
        address = dict()
        if neighbor_id:
            if isinstance(neighbor_id, list):
                for each_neighbor in neighbor_id:
                    address[each_neighbor] = self.get_neighbors(each_neighbor)
                return address
            else:
                return self.neighbors[neighbor_id]
        else:
            # Get all neighbors
            return self.neighbors

    def get_num_neighbors(self):
        return len(self.neighbors)

    def sample_neighbor(self, n_exor):
        exor_idx = self.neighbors.keys()
        assert len(exor_idx) >= n_exor, f"Executor ({len(exor_idx)}) is not enough for {n_exor}"
        exor_choice = np.random.choice(exor_idx, size=n_exor, replace=False)
        return exor_choice

    """
    Functions used to receive messages
    """
    def receive(self):
        message = self.queue_receive.get(block=True)
        return message

    """
    Functions used to send messages and models
    """
    def send_file(self, msg_type, state, path_file, executor_choice, group_client):
        # Load files here once
        chunk = get_file_chunks(path_file)
        if not isinstance(executor_choice, list):
            executor_choice = list(executor_choice)

        for executor_id, client_idx in zip(executor_choice, group_client):
            if executor_id in self.neighbors.keys():
                executor_address = self.neighbors[executor_id]
                self.queue_send.put((msg_type, executor_address, (client_idx, state, chunk)))
            else:
                executor_idle = list(set(self.neighbors.keys())-set(executor_choice))
                if len(executor_idle) > 0:
                    executor_id = np.random.choice(executor_idle, 1, replace=False)[0]
                    executor_address = self.neighbors[executor_id]
                    self.queue_send.put((msg_type, executor_address, (client_idx, state, chunk)))
                else:
                    logger.info(f"Skipping send message {msg_type} to executor_id #{executor_id}")

    def send(self, message):
        receiver = message.receiver
        msg_type = message.msg_type

        if receiver is None:
            receiver = list(self.neighbors.keys())
        elif not isinstance(receiver, list):
            receiver = list(receiver)

        for receiver_id in receiver:
            receiver_address = self.neighbors[receiver_id]
            self.queue_send.put((msg_type, receiver_address, message))
