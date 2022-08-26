from abc import ABC, abstractmethod


class Aggregator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def aggregate(self, agg_info):
        pass


class NoCommunicationAggregator(Aggregator):
    """"Clients do not communicate. Each client work locally
    """
    def aggregate(self, agg_info):
        # do nothing
        return {}
