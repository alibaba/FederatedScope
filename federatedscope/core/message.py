import json
import numpy as np
from federatedscope.core.proto import gRPC_comm_manager_pb2


class Message(object):
    """
    The data exchanged during an FL course are abstracted as 'Message' in
    FederatedScope.
    A message object includes:
        msg_type: The type of message, which is used to trigger the
        corresponding handlers of server/client
        sender: The sender's ID
        receiver: The receiver's ID
        state: The training round of the message, which is determined by
        the sender and used to filter out the outdated messages.
        strategy: redundant attribute
    """
    def __init__(self,
                 msg_type=None,
                 sender=0,
                 receiver=0,
                 state=0,
                 content=None,
                 strategy=None):
        self._msg_type = msg_type
        self._sender = sender
        self._receiver = receiver
        self._state = state
        self._content = content
        self._strategy = strategy

    @property
    def msg_type(self):
        return self._msg_type

    @msg_type.setter
    def msg_type(self, value):
        self._msg_type = value

    @property
    def sender(self):
        return self._sender

    @sender.setter
    def sender(self, value):
        self._sender = value

    @property
    def receiver(self):
        return self._receiver

    @receiver.setter
    def receiver(self, value):
        self._receiver = value

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = value

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, value):
        self._strategy = value

    def transform_to_list(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            return [self.transform_to_list(each_x) for each_x in x]
        elif isinstance(x, dict):
            for key in x.keys():
                x[key] = self.transform_to_list(x[key])
            return x
        else:
            if hasattr(x, 'tolist'):
                return x.tolist()
            else:
                return x

    def msg_to_json(self, to_list=False):
        if to_list:
            self.content = self.transform_to_list(self.content)

        json_msg = {
            'msg_type': self.msg_type,
            'sender': self.sender,
            'receiver': self.receiver,
            'state': self.state,
            'content': self.content,
            'strategy': self.strategy,
        }
        return json.dumps(json_msg)

    def json_to_msg(self, json_string):
        json_msg = json.loads(json_string)
        self.msg_type = json_msg['msg_type']
        self.sender = json_msg['sender']
        self.receiver = json_msg['receiver']
        self.state = json_msg['state']
        self.content = json_msg['content']
        self.strategy = json_msg['strategy']

    def create_by_type(self, value, nested=False):
        if isinstance(value, dict):
            m_dict = gRPC_comm_manager_pb2.mDict()
            for key in value.keys():
                m_dict.dict_value[key].MergeFrom(
                    self.create_by_type(value[key], nested=True))
            if nested:
                msg_value = gRPC_comm_manager_pb2.MsgValue()
                msg_value.dict_msg.MergeFrom(m_dict)
                return msg_value
            else:
                return m_dict
        elif isinstance(value, list) or isinstance(value, tuple):
            m_list = gRPC_comm_manager_pb2.mList()
            for each in value:
                m_list.list_value.append(self.create_by_type(each,
                                                             nested=True))
            if nested:
                msg_value = gRPC_comm_manager_pb2.MsgValue()
                msg_value.list_msg.MergeFrom(m_list)
                return msg_value
            else:
                return m_list
        else:
            m_single = gRPC_comm_manager_pb2.mSingle()
            if type(value) in [int, np.int32]:
                m_single.int_value = value
            elif type(value) in [str]:
                m_single.str_value = value
            elif type(value) in [float, np.float32]:
                m_single.float_value = value
            else:
                raise ValueError(
                    'The data type {} has not been supported.'.format(
                        type(value)))

            if nested:
                msg_value = gRPC_comm_manager_pb2.MsgValue()
                msg_value.single_msg.MergeFrom(m_single)
                return msg_value
            else:
                return m_single

    def build_msg_value(self, value):
        msg_value = gRPC_comm_manager_pb2.MsgValue()

        if isinstance(value, list) or isinstance(value, tuple):
            msg_value.list_msg.MergeFrom(self.create_by_type(value))
        elif isinstance(value, dict):
            msg_value.dict_msg.MergeFrom(self.create_by_type(value))
        else:
            msg_value.single_msg.MergeFrom(self.create_by_type(value))

        return msg_value

    def transform(self, to_list=False):
        if to_list:
            self.content = self.transform_to_list(self.content)

        splited_msg = gRPC_comm_manager_pb2.MessageRequest()  # map/dict
        splited_msg.msg['sender'].MergeFrom(self.build_msg_value(self.sender))
        splited_msg.msg['receiver'].MergeFrom(
            self.build_msg_value(self.receiver))
        splited_msg.msg['state'].MergeFrom(self.build_msg_value(self.state))
        splited_msg.msg['msg_type'].MergeFrom(
            self.build_msg_value(self.msg_type))
        splited_msg.msg['content'].MergeFrom(self.build_msg_value(
            self.content))
        return splited_msg

    def _parse_msg(self, value):
        if isinstance(value, gRPC_comm_manager_pb2.MsgValue) or isinstance(
                value, gRPC_comm_manager_pb2.mSingle):
            return self._parse_msg(getattr(value, value.WhichOneof("type")))
        elif isinstance(value, gRPC_comm_manager_pb2.mList):
            return [self._parse_msg(each) for each in value.list_value]
        elif isinstance(value, gRPC_comm_manager_pb2.mDict):
            return {
                k: self._parse_msg(value.dict_value[k])
                for k in value.dict_value
            }
        else:
            return value

    def parse(self, received_msg):
        self.sender = self._parse_msg(received_msg['sender'])
        self.receiver = self._parse_msg(received_msg['receiver'])
        self.msg_type = self._parse_msg(received_msg['msg_type'])
        self.state = self._parse_msg(received_msg['state'])
        self.content = self._parse_msg(received_msg['content'])

    def count_bytes(self):
        """
            calculate the message bytes to be sent/received
        :return: tuple of bytes of the message to be sent and received
        """
        from pympler import asizeof
        download_bytes = asizeof.asizeof(self.content)
        upload_cnt = len(self.receiver) if isinstance(self.receiver,
                                                      list) else 1
        upload_bytes = download_bytes * upload_cnt
        return download_bytes, upload_bytes
