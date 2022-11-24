import abc
from federatedscope.core.workers import Worker


class BaseServer(Worker):
    def __init__(self, ID, state, config, model, strategy):
        super(BaseServer, self).__init__(ID, state, config, model, strategy)
        self.msg_handlers = dict()
        self.msg_handlers_str = dict()

    def register_handlers(self, msg_type, callback_func, send_msg=[None]):
        """
        To bind a message type with a handling function.

        Arguments:
            msg_type (str): The defined message type
            callback_func: The handling functions to handle the received \
                message
        """
        self.msg_handlers[msg_type] = callback_func
        self.msg_handlers_str[msg_type] = (callback_func.__name__, send_msg)

    def _register_default_handlers(self):
        """
        Register default handler dic to handle message, which includes \
        sender, receiver, state, and content. More detail can be found in \
        ``federatedscope.core.message``.

        Note:
          the default handlers to handle messages and related callback \
          function are shown below:
            ============================ ==================================
            Message type                 Callback function
            ============================ ==================================
            ``join_in``                  ``callback_funcs_for_join_in()``
            ``join_in_info``             ``callback_funcs_for_join_in()``
            ``model_para``               ``callback_funcs_model_para()``
            ``metrics``                  ``callback_funcs_for_metrics``
            ============================ ==================================
        """
        self.register_handlers('join_in', self.callback_funcs_for_join_in, [
            'assign_client_id', 'ask_for_join_in_info', 'address', 'model_para'
        ])
        self.register_handlers('join_in_info', self.callback_funcs_for_join_in,
                               ['address', 'model_para'])
        self.register_handlers('model_para', self.callback_funcs_model_para,
                               ['model_para', 'evaluate', 'finish'])
        self.register_handlers('metrics', self.callback_funcs_for_metrics,
                               ['converged'])

    @abc.abstractmethod
    def run(self):
        """
        To start the FL course, listen and handle messages (for distributed \
        mode).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_model_para(self, message):
        """
        The handling function for receiving model parameters, which triggers \
        ``check_and_move_on`` (perform aggregation when enough feedback has \
        been received). This handling function is widely used in various FL \
        courses.

        Arguments:
            message: The received message.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_for_join_in(self, message):
        """
        The handling function for receiving the join in information. The \
        server might request for some information (such as \
        ``num_of_samples``) if necessary, assign IDs for the servers. \
        If all the clients have joined in, the training process will be \
        triggered.

        Arguments:
            message: The received message
        """
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_for_metrics(self, message):
        """
        The handling function for receiving the evaluation results, \
        which triggers ``check_and_move_on`` (perform aggregation when \
        enough feedback has been received).

        Arguments:
            message: The received message
        """
        raise NotImplementedError
