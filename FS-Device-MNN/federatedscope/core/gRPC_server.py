import os

import google.protobuf.empty_pb2
import torch

import MNN.expr as F
import numpy as np

from copy import deepcopy
from federatedscope.core.message import Message
from federatedscope.core.proto import gRPC_comm_manager_pb2, \
    gRPC_comm_manager_pb2_grpc

FIRST = True

def trans_dict(model_dict, model_type):
    if model_type.lower() == "lr":
        keys_map = {
            "TrainableParam2": "fc.weight",
            "TrainableParam1": "fc.bias"
        }
        new_dict = dict()
        for key, value in keys_map.items():
            if value in ["fc.weight"]:
                array = deepcopy(np.squeeze(model_dict[key].read()))
            else:
                array = deepcopy(model_dict[key].read())
            new_dict[value] = torch.from_numpy(array)

        return new_dict
    elif model_type.lower() == "convnet2":
        # for ConvNet2
        if len(model_dict.keys()) == 8:
            keys_map = {
                "TrainableParam8": "conv1.weight",
                "TrainableParam7": "conv1.bias",
                "TrainableParam6": "conv2.weight",
                "TrainableParam5": "conv2.bias",
                "TrainableParam4": "fc1.weight",
                "TrainableParam3": "fc1.bias",
                "TrainableParam2": "fc2.weight",
                "TrainableParam1": "fc2.bias"
            }
        else:
            keys_map = {
                "TrainableParam6": "conv1.weight",
                "TrainableParam5": "conv1.bias",
                "TrainableParam4": "conv2.weight",
                "TrainableParam3": "conv2.bias",
                "TrainableParam2": "fc1.weight",
                "TrainableParam1": "fc1.bias",
            }
        new_dict = dict()
        for key, value in keys_map.items():
            if value in ["fc1.weight", "fc2.weight"]:
                array = deepcopy(np.squeeze(model_dict[key].read()))
            else:
                array = deepcopy(model_dict[key].read())
            new_dict[value] = torch.from_numpy(array)

        return new_dict

    elif model_type.lower() == "convnet5":
        if len(model_dict.keys()) == 14:
            keys_map = {
                "TrainableParam1": "fc2.bias",
                "TrainableParam2": "fc2.weight",
                "TrainableParam3": "fc1.bias",
                "TrainableParam4": "fc1.weight",
                "TrainableParam5": "conv5.bias",
                "TrainableParam6": "conv5.weight",
                "TrainableParam7": "conv4.bias",
                "TrainableParam8": "conv4.weight",
                "TrainableParam9": "conv3.bias",
                "TrainableParam10": "conv3.weight",
                "TrainableParam11": "conv2.bias",
                "TrainableParam12": "conv2.weight",
                "TrainableParam13": "conv1.bias",
                "TrainableParam14": "conv1.weight"
            }
            # keys_map = {
            #     "TrainableParam1": "fc2.bias",
            #     "TrainableParam2": "fc2.weight",
            #     "TrainableParam3": "fc1.bias",
            #     "TrainableParam4": "fc1.weight",
            #     "TrainableParam5": "bn5.weight",
            #     "TrainableParam6": "bn5.bias",
            #     "Const7": "bn5.running_var",
            #     "Const8": "bn5.running_mean",
            #     "TrainableParam9": "conv5.bias",
            #     "TrainableParam10": "conv5.weight",
            #     "TrainableParam11": "bn4.weight",
            #     "TrainableParam12": "bn4.bias",
            #     "Const13": "bn4.running_var",
            #     "Const14": "bn4.running_mean",
            #     "TrainableParam15": "conv4.bias",
            #     "TrainableParam16": "conv4.weight",
            #     "TrainableParam17": "bn3.weight",
            #     "TrainableParam18": "bn3.bias",
            #     "Const19": "bn3.running_var",
            #     "Const20": "bn3.running_mean",
            #     "TrainableParam21": "conv3.bias",
            #     "TrainableParam22": "conv3.weight",
            #     "TrainableParam23": "bn1.weight",
            #     "TrainableParam24": "bn1.bias",
            #     "Const25": "bn2.running_var",
            #     "Const26": "bn2.running_mean",
            #     "TrainableParam27": "conv2.bias",
            #     "TrainableParam28": "conv2.weight",
            #     "TrainableParam29": "bn1.weight",
            #     "TrainableParam30": "bn1.bias",
            #     "Const31": "bn1.running_var",
            #     "Const32": "bn1.running_mean",
            #     "TrainableParam33": "conv1.bias",
            #     "TrainableParam34": "conv1.weight"
            # }
        else:
            keys_map = {
                "TrainableParam1": "fc1.bias",
                "TrainableParam2": "fc1.weight",
                "TrainableParam3": "conv5.bias",
                "TrainableParam4": "conv5.weight",
                "TrainableParam5": "conv4.bias",
                "TrainableParam6": "conv4.weight",
                "TrainableParam7": "conv3.bias",
                "TrainableParam8": "conv3.weight",
                "TrainableParam9": "conv2.bias",
                "TrainableParam10": "conv2.weight",
                "TrainableParam11": "conv1.bias",
                "TrainableParam12": "conv1.weight"
            }
            # keys_map = {
            #     "TrainableParam3": "fc1.bias",
            #     "TrainableParam4": "fc1.weight",
            #     "TrainableParam5": "bn5.weight",
            #     "TrainableParam6": "bn5.bias",
            #     "Const7": "bn5.running_var",
            #     "Const8": "bn5.running_mean",
            #     "TrainableParam9": "conv5.bias",
            #     "TrainableParam10": "conv5.weight",
            #     "TrainableParam11": "bn4.weight",
            #     "TrainableParam12": "bn4.bias",
            #     "Const13": "bn4.running_var",
            #     "Const14": "bn4.running_mean",
            #     "TrainableParam15": "conv4.bias",
            #     "TrainableParam16": "conv4.weight",
            #     "TrainableParam17": "bn3.weight",
            #     "TrainableParam18": "bn3.bias",
            #     "Const19": "bn3.running_var",
            #     "Const20": "bn3.running_mean",
            #     "TrainableParam21": "conv3.bias",
            #     "TrainableParam22": "conv3.weight",
            #     "TrainableParam23": "bn1.weight",
            #     "TrainableParam24": "bn1.bias",
            #     "Const25": "bn2.running_var",
            #     "Const26": "bn2.running_mean",
            #     "TrainableParam27": "conv2.bias",
            #     "TrainableParam28": "conv2.weight",
            #     "TrainableParam29": "bn1.weight",
            #     "TrainableParam30": "bn1.bias",
            #     "Const31": "bn1.running_var",
            #     "Const32": "bn1.running_mean",
            #     "TrainableParam33": "conv1.bias",
            #     "TrainableParam34": "conv1.weight",
            # }

        new_dict = dict()
        for key, value in keys_map.items():
            array = deepcopy(np.squeeze(model_dict[key].read()))
            new_dict[value] = torch.from_numpy(array)
        return new_dict


class ServiceFunc(gRPC_comm_manager_pb2_grpc.gRPCComServeFuncServicer):
    def __init__(self, save_path, queue, model_type):
        self.save_path = save_path
        self.queue = queue
        self.model_type = model_type

    def uploadMnnModel(self, request, context):
        path_file = os.path.join(self.save_path, f"CLIENT-{request.info.sender}_STATE-{request.info.state}.mnn")
        with open(path_file, 'wb') as f:
            f.write(request.chunk)
        model_dict = trans_dict(F.load_as_dict(path_file), self.model_type)
        # Delete file to avoid storing too many models
        os.remove(path_file)

        received_msg = Message(msg_type="model_para",
                               sender=request.info.sender,
                               state=request.info.state,
                               timestamp=request.info.timestamp,
                               content=(request.info.n_sample, model_dict))

        self.queue.put(received_msg, block=False)
        return google.protobuf.empty_pb2.Empty()

    def sendMessage(self, request, context):
        message = Message()
        message.parse(request.msg)
        self.queue.put(message, block=False)
        return gRPC_comm_manager_pb2.MessageResponse(msg='ACK')