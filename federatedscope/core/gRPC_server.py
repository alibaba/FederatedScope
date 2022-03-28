import queue

from federatedscope.core.proto import gRPC_comm_manager_pb2, gRPC_comm_manager_pb2_grpc


class gRPCComServeFunc(gRPC_comm_manager_pb2_grpc.gRPCComServeFuncServicer):
    def __init__(self):
        self.msg_queue = queue.Queue()

    def sendMessage(self, request, context):
        self.msg_queue.put(request)

        return gRPC_comm_manager_pb2.MessageResponse(msg='ACK')

    def receive(self):
        while self.msg_queue.empty():
            continue
        msg = self.msg_queue.get()
        return msg
