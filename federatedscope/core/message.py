import sys
import json


class Message(object):
    """
    The data exchanged during an FL course are abstracted as 'Message' in FederatedScope.
    A message object includes:
        msg_type: The type of message, which is used to trigger the corresponding handlers of server/client
        sender: The sender's ID
        receiver: The receiver's ID
        state: The training round of the message, which is determined by the sender and used to filter out the outdated messages.
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
