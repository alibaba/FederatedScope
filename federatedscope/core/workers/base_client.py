import abc
from federatedscope.core.workers.base_worker import Worker


class BaseClient(Worker):
    def __init__(self, ID, state, config, model, strategy):
        super(BaseClient, self).__init__(ID, state, config, model, strategy)
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
            ``assign_client_id``         ``callback_funcs_for_assign_id()``
            ``ask_for_join_in_info``     ``callback_funcs_for_join_in_info()``
            ``address``                  ``callback_funcs_for_address()``
            ``model_para``               ``callback_funcs_for_model_para()``
            ``ss_model_para``            ``callback_funcs_for_model_para()``
            ``evaluate``                 ``callback_funcs_for_evaluate()``
            ``finish``                   ``callback_funcs_for_finish()``
            ``converged``                ``callback_funcs_for_converged()``
            ============================ ==================================
        """
        self.register_handlers('assign_client_id',
                               self.callback_funcs_for_assign_id, [None])
        self.register_handlers('ask_for_join_in_info',
                               self.callback_funcs_for_join_in_info,
                               ['join_in_info'])
        self.register_handlers('address', self.callback_funcs_for_address,
                               [None])
        self.register_handlers('model_para',
                               self.callback_funcs_for_model_para,
                               ['model_para', 'ss_model_para'])
        self.register_handlers('ss_model_para',
                               self.callback_funcs_for_model_para,
                               ['ss_model_para', 'model_para'])
        self.register_handlers('evaluate', self.callback_funcs_for_evaluate,
                               ['metrics'])
        self.register_handlers('finish', self.callback_funcs_for_finish,
                               [None])
        self.register_handlers('converged', self.callback_funcs_for_converged,
                               [None])

    @abc.abstractmethod
    def run(self):
        """
        To listen to the message and handle them accordingly (used for \
        distributed mode)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_for_model_para(self, message):
        """
        The handling function for receiving model parameters, \
        which triggers the local training process. \
        This handling function is widely used in various FL courses.

        Arguments:
            message: The received message
        """
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_for_assign_id(self, message):
        """
        The handling function for receiving the client_ID assigned by the \
        server (during the joining process), which is used in the \
        distributed mode.

        Arguments:
            message: The received message
        """
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_for_join_in_info(self, message):
        """
        The handling function for receiving the request of join in \
        information (such as ``batch_size``, ``num_of_samples``) during \
        the joining process.

        Arguments:
            message: The received message
        """
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_for_address(self, message):
        """
        The handling function for receiving other clients' IP addresses, \
        which is used for constructing a complex topology

        Arguments:
            message: The received message
        """
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_for_evaluate(self, message):
        """
        The handling function for receiving the request of evaluating

        Arguments:
            message: The received message
        """
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_for_finish(self, message):
        """
        The handling function for receiving the signal of finishing the FL \
        course.

        Arguments:
            message: The received message
        """
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_for_converged(self, message):
        """
        The handling function for receiving the signal that the FL course \
        converged

        Arguments:
            message: The received message
        """
        raise NotImplementedError
