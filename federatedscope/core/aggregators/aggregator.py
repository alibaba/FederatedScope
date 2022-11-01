from abc import ABC, abstractmethod


class Aggregator(ABC):
    """
    Abstract class of Aggregator.
    """
    def __init__(self):
        pass

    @abstractmethod
    def aggregate(self, agg_info):
        """
        Aggregation function.

        Args:
            agg_info: information to be aggregated.
        """
        pass


class NoCommunicationAggregator(Aggregator):
    """Clients do not communicate. Each client work locally
    """
    def aggregate(self, agg_info):
        """
        Aggregation function.

        Args:
            agg_info: information to be aggregated.
        """
        # do nothing
        return {}
