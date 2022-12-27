import queue
import time
from collections import deque

from federatedscope.core.proto import gRPC_comm_manager_pb2, \
    gRPC_comm_manager_pb2_grpc


class gRPCComServeFunc(gRPC_comm_manager_pb2_grpc.gRPCComServeFuncServicer):
    def __init__(self):
        self.msg_queue = deque()

    def sendMessage(self, request, context):
        self.init_time = time.time()
        print('message is received')
        self.msg_queue.append(request)

        return gRPC_comm_manager_pb2.MessageResponse(msg='ACK')

    def receive(self):
        while len(self.msg_queue) == 0:
            continue
        msg = self.msg_queue.popleft()
        print('message is pop ',time.time() - self.init_time)
        return msg
