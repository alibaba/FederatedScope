import grpc
from concurrent import futures
import logging
import math

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.proto import gRPC_comm_manager_pb2, \
    gRPC_comm_manager_pb2_grpc
from federatedscope.core.gRPC_server import gRPCComServeFunc
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)

# class StandaloneCommManager(object):
#     """
#     The communicator used for standalone mode
#     """
#     def __init__(self, comm_queue, monitor=None):
#         self.comm_queue = comm_queue
#         self.neighbors = dict()
#         self.monitor = monitor  # used to track the communication related
#         # metrics

#     def receive(self):
#         # we don't need receive() in standalone
#         pass

#     def add_neighbors(self, neighbor_id, address=None):
#         self.neighbors[neighbor_id] = address

#     def get_neighbors(self, neighbor_id=None):
#         address = dict()
#         if neighbor_id:
#             if isinstance(neighbor_id, list):
#                 for each_neighbor in neighbor_id:
#                     address[each_neighbor] = self.get_neighbors(each_neighbor)
#                 return address
#             else:
#                 return self.neighbors[neighbor_id]
#         else:
#             # Get all neighbors
#             return self.neighbors

#     def send(self, message):
#         self.comm_queue.append(message)
#         download_bytes, upload_bytes = message.count_bytes()
#         self.monitor.track_upload_bytes(upload_bytes)

class StandaloneClientCommManager(object):
    """
    The communicator used for standalone mode
    """
    def __init__(self, send_channel, monitor=None):
        self.send_channel = send_channel
        self.neighbors = dict()
        self.monitor = monitor  # used to track the communication related
        # metrics

    def receive(self):
        # we don't need receive() in standalone
        pass

    def add_neighbors(self, neighbor_id, address=None):
        self.neighbors[neighbor_id] = address

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

    def send(self, message):
        logger.info(f"client send message {message.msg_type}")
        self.send_channel.put(message)
        download_bytes, upload_bytes = message.count_bytes()
        self.monitor.track_upload_bytes(upload_bytes)


class StandaloneServerCommManager(object):
    """
    The communicator used for standalone mode
    """
    def __init__(self, client_num, channels, monitor=None):
        self.client_num = client_num
        self.process_num = len(channels)
        self.part_size = math.ceil(self.client_num / self.process_num)
        self.send_channel = channels
        self.neighbors = dict()
        self.monitor = monitor  # used to track the communication related
        # metrics

    def receive(self):
        # we don't need receive() in standalone
        pass

    def add_neighbors(self, neighbor_id, address=None):
        self.neighbors[neighbor_id] = address

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

    def send(self, message):
        download_bytes, upload_bytes = message.count_bytes()
        receiver = message.receiver
        if receiver is not None:
            if not isinstance(receiver, list):
                receiver = [receiver]
            for process_id in range(0, self.process_num):
                for each_receiver in receiver:
                    if each_receiver in self.neighbors:
                        logger.info(f"server send message to {each_receiver} in client runner {(each_receiver - 1) // self.part_size}")
                        self.send_channel[process_id].put(message)
                        self.monitor.track_upload_bytes(upload_bytes)
                        break
        else:
            logger.error(f"wrong message type in standalone mode")


class gRPCCommManager(object):
    """
        The implementation of gRPCCommManager is referred to the tutorial on
        https://grpc.io/docs/languages/python/
    """
    def __init__(self, host='0.0.0.0', port='50050', client_num=2):
        self.host = host
        self.port = port
        options = [
            ("grpc.max_send_message_length",
             global_cfg.distribute.grpc_max_send_message_length),
            ("grpc.max_receive_message_length",
             global_cfg.distribute.grpc_max_receive_message_length),
            ("grpc.enable_http_proxy",
             global_cfg.distribute.grpc_enable_http_proxy),
        ]
        self.server_funcs = gRPCComServeFunc()
        self.grpc_server = self.serve(max_workers=client_num,
                                      host=host,
                                      port=port,
                                      options=options)
        self.neighbors = dict()
        self.monitor = None  # used to track the communication related metrics

    def serve(self, max_workers, host, port, options):
        """
        This function is referred to
        https://grpc.io/docs/languages/python/basics/#starting-the-server
        """
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=options)
        gRPC_comm_manager_pb2_grpc.add_gRPCComServeFuncServicer_to_server(
            self.server_funcs, server)
        server.add_insecure_port("{}:{}".format(host, port))
        server.start()

        return server

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

    def _send(self, receiver_address, message):
        def _create_stub(receiver_address):
            """
            This part is referred to
            https://grpc.io/docs/languages/python/basics/#creating-a-stub
            """
            channel = grpc.insecure_channel(receiver_address,
                                            options=(('grpc.enable_http_proxy',
                                                      0), ))
            stub = gRPC_comm_manager_pb2_grpc.gRPCComServeFuncStub(channel)
            return stub, channel

        stub, channel = _create_stub(receiver_address)
        request = message.transform(to_list=True)
        try:
            stub.sendMessage(request)
        except grpc._channel._InactiveRpcError:
            pass
        channel.close()

    def send(self, message):
        receiver = message.receiver
        if receiver is not None:
            if not isinstance(receiver, list):
                receiver = [receiver]
            for each_receiver in receiver:
                if each_receiver in self.neighbors:
                    receiver_address = self.neighbors[each_receiver]
                    self._send(receiver_address, message)
        else:
            for each_receiver in self.neighbors:
                receiver_address = self.neighbors[each_receiver]
                self._send(receiver_address, message)

    def receive(self):
        received_msg = self.server_funcs.receive()
        message = Message()
        message.parse(received_msg.msg)
        return message
