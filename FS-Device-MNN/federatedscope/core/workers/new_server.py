from federatedscope.core.auxiliaries.aggregator_builder import get_aggregator
from federatedscope.core.message import Message


class EVENTTYPE:
    BLOCK = "BLOCK"
    NON_BLOCK = "NON_BLOCK"


class JoinInEvent(object):
    event_type = EVENTTYPE.BLOCK
    msg_type = "join_in"

    def __init__(self, event_next=None):
        self.event_next = event_next

    def trigger(self):
        if self.event_type == EVENTTYPE.NON_BLOCK and self.event_next is not None:
            self.event_next.trigger()
        # Do some init things
        pass

    def handle(self, msg):
        ClientManager().register_client(msg.content)

    def finished(self):
        pass


class BroadcastEvent:
    event_type = EVENTTYPE.BLOCK
    msg_type = "model_para"

    def __init__(self, model, device, event_next=None):
        self.event_next = event_next

        self.aggregator = get_aggregator(self._cfg.federate.method,
                                         model=model,
                                         device=device,
                                         online=self._cfg.federate.online_aggr,
                                         config=self._cfg)

        self.para_buffer = list()  # not online aggregation

    def trigger(self, model_dict, *args, **kwargs):
        # Setup next event is the current one is non-block
        if self.event_type == EVENTTYPE.NON_BLOCK and self.event_next is not None:
            self.event_next.trigger()

        self.send(Message(
            msg_type=self.msg_type,
            sender=0,
            content=model_dict
        ))

    def handle(self, msg):
        model_dict = msg.content
        # pack into para_buffer
        self.para_buffer.append(model_dict)
        # collect enough msgs
        if len(self.para_buffer) == self.target_num:
            self.finished()

    def finished(self):
        # Perform aggregation
        aggregated_model = self.aggregator.aggregate(self.para_buffer)
        # The current event is over, setup next event
        if self.event_type == EVENTTYPE.BLOCK and self.event_next is not None:
            self.event_next.trigger()

        return aggregated_model


class TestEvent:
    event_type = EVENTTYPE.NON_BLOCK
    msg_type = "model_para"

    def __init__(self, event_next=None):
        self.event_next = event_next

    def trigger(self):
        if self.event_type == EVENTTYPE.NON_BLOCK and self.event_next is not None:
            self.event_next.trigger()

    def finished(self):
        if self.event_type == EVENTTYPE.BLOCK and self.event_next is not None:
            self.event_next.trigger()


class ClientObj(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setattr__


class ClientManager:
    """
    Achieved in Singleton
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.clients = list()

    def get_all_clients(self):
        return self.clients

    def register_client(self, client) -> int:
        if client not in self.clients:
            self.clients.append(client)
            return len(self.clients)  # return new id
        else:
            return -1  # client has already registered


class Server(object):
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device="cpu",
                 **kwargs):
        self._cfg = config

        self.ID = ID
        self.device = device

        self.model = model

        self.events = [
            JoinInEvent,
            [
                total_round_num,  # repeat times
                BroadcastEvent,
                TestEvent,
            ]
        ]

    def start(self):
        for event in self.events:
            if isinstance(list):
                for i, sub_event in enumerate(event):
                    if i == 0 and sub_event.__class__ == int:
                        # repeat times
                        for k in range(sub_event):
                            pass
