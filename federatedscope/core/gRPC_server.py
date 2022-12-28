import queue
import time
from collections import deque
from multiprocessing import Queue
from federatedscope.core.message import Message

from federatedscope.core.proto import gRPC_comm_manager_pb2, \
    gRPC_comm_manager_pb2_grpc


class gRPCComServeFunc(gRPC_comm_manager_pb2_grpc.gRPCComServeFuncServicer):
    def __init__(self):
        self.msg_queue = Queue()

    def sendMessage(self, request, context):
        self.init_time = time.time()
        print('message is received ', time.asctime( time.localtime(time.time()) ))
        message = Message()
        message.parse(request.msg)
        self.msg_queue.put(message)

        return gRPC_comm_manager_pb2.MessageResponse(msg='ACK')

    def receive(self):
        while self.msg_queue.empty():
            continue
        msg = self.msg_queue.get()
        print('message is pop ',time.asctime( time.localtime(time.time()) ))
        return msg
