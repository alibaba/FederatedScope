# Copyright [FedML] [Chaoyang He, Salman Avestimehr]
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import grpc
from concurrent import futures

from federatedscope.config import cfg
from federatedscope.core.proto import gRPC_comm_manager_pb2, gRPC_comm_manager_pb2_grpc
from federatedscope.core.gRPC_server import gRPCComServeFunc
from federatedscope.core.message import Message


class StandaloneCommManager(object):
    def __init__(self, comm_queue):
        self.comm_queue = comm_queue
        self.neighbors = dict()

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
        self.comm_queue.append(message)


class gRPCCommManager(object):
    """
        The implementation of gRPCCommManager is referred to https://github.com/FedML-AI/FedML/tree/master/fedml_core/distributed/communication/gRPC
    """
    def __init__(self, host='0.0.0.0', port='50050', client_num=2):
        grpc_opts = [
            ("grpc.max_send_message_length",
             cfg.distribute.grpc_max_send_message_length),
            ("grpc.max_receive_message_length",
             cfg.distribute.grpc_max_receive_message_length),
            ("grpc.enable_http_proxy", cfg.distribute.grpc_enable_http_proxy),
        ]
        self.grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=client_num),
            options=grpc_opts)
        self.servicer = gRPCComServeFunc()
        gRPC_comm_manager_pb2_grpc.add_gRPCComServeFuncServicer_to_server(
            servicer=self.servicer, server=self.grpc_server)
        self.grpc_server.add_insecure_port("{}:{}".format(host, port))
        self.host = host
        self.port = port
        self.grpc_server.start()

        self.neighbors = dict()

    def add_neighbors(self, neighbor_id, address):
        self.neighbors[neighbor_id] = '{}:{}'.format(address['host'],
                                                     address['port'])

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
            #Get all neighbors
            return self.neighbors

    def _send(self, receiver_address, message):
        channel = grpc.insecure_channel(receiver_address,
                                        options=(('grpc.enable_http_proxy',
                                                  0), ))
        stub = gRPC_comm_manager_pb2_grpc.gRPCComServeFuncStub(channel)
        request = message.msg_to_json(to_list=True)
        stub.sendMessage(gRPC_comm_manager_pb2.MessageRequest(msg=request))
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
        received_msg = self.servicer.receive()
        message = Message()
        message.json_to_msg(received_msg.msg)
        return message
