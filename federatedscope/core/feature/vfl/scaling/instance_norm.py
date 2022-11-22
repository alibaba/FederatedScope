import types
import logging
import numpy as np

from federatedscope.core.message import Message

logger = logging.getLogger(__name__)


def wrap_instance_norm_for_server(worker):
    """
    This function is to perform instance norm vfl tabular data for server.
    Args:
        worker: ``federatedscope.core.workers.Worker`` to be wrapped

    Returns:
        Wrap vfl server with instance norm.
    """
    def trigger_for_start(self):
        if self.check_client_join_in():
            self.broadcast_client_address()
            # Ask for instance statistic
            self.msg_buffer['instance_mean'] = {}
            self.msg_buffer['instance_var'] = {}
            logger.info('Start to execute instance norm.')
            self.comm_manager.send(
                Message(msg_type='ask_for_instance_mean',
                        sender=self.ID,
                        receiver=list(self.comm_manager.neighbors.keys()),
                        timestamp=self.cur_timestamp))

    def callback_func_for_instance_mean(self, message: Message):
        stat = message.content
        sender = message.sender
        self.msg_buffer['instance_mean'][sender] = stat

        if len(self.msg_buffer['instance_mean'].keys()) == self._client_num:
            # Aggregate mean
            instance_mean = {}
            for split in self.msg_buffer['instance_mean'][sender]:
                dim = 0
                feat_sum = np.zeros_like(
                    self.msg_buffer['instance_mean'][sender][split]['mean'])
                for stat in self.msg_buffer['instance_mean'].values():
                    feat_sum += stat[split]['mean'] * stat[split]['dim']
                    dim += stat[split]['dim']
                instance_mean[split] = feat_sum / dim
            self.comm_manager.send(
                Message(msg_type='instance_mean',
                        sender=self.ID,
                        receiver=list(self.comm_manager.neighbors.keys()),
                        timestamp=self.cur_timestamp,
                        content=instance_mean))

    def callback_func_for_instance_var(self, message: Message):
        stat = message.content
        sender = message.sender
        self.msg_buffer['instance_var'][sender] = stat

        if len(self.msg_buffer['instance_var'].keys()) == self._client_num:
            # Aggregate var
            instance_var = {}
            for split in self.msg_buffer['instance_var'][sender]:
                dim = 0
                feat_sum = np.zeros_like(
                    self.msg_buffer['instance_var'][sender][split]['var'])
                for stat in self.msg_buffer['instance_var'].values():
                    feat_sum += stat[split]['var'] * stat[split]['dim']
                    dim += stat[split]['dim']
                instance_var[split] = feat_sum / dim
            self.comm_manager.send(
                Message(msg_type='instance_var',
                        sender=self.ID,
                        receiver=list(self.comm_manager.neighbors.keys()),
                        timestamp=self.cur_timestamp,
                        content=instance_var))

            # Start to train
            self.broadcast_model_para()

    # Bind method to instance
    worker.trigger_for_start = types.MethodType(trigger_for_start, worker)
    worker.callback_func_for_instance_mean = types.MethodType(
        callback_func_for_instance_mean, worker)
    worker.callback_func_for_instance_var = types.MethodType(
        callback_func_for_instance_var, worker)

    # Register handlers functions
    worker.register_handlers('instance_mean',
                             worker.callback_func_for_instance_mean)
    worker.register_handlers('instance_var',
                             worker.callback_func_for_instance_var)
    return worker


def wrap_instance_norm_for_client(worker):
    """
    This function is to perform instance norm vfl tabular data for client.
    Args:
        worker: ``federatedscope.core.workers.Worker`` to be wrapped

    Returns:
        Wrap vfl client with instance norm.
    """
    def callback_func_for_ask_for_mean(self, message: Message):
        sender = message.sender
        stat = {}
        # Calculate mean
        for split in ['train_data', 'val_data', 'test_data']:
            if hasattr(self.data, split):
                split_data = getattr(self.data, split)
                if split_data is not None and 'x' in split_data:
                    stat[split] = {
                        'mean': np.mean(split_data['x'], axis=1),
                        'dim': np.shape(split_data['x'])[1],
                    }

        # Send mean
        self.comm_manager.send(
            Message(msg_type='instance_mean',
                    sender=self.ID,
                    receiver=sender,
                    content=stat))

    def callback_func_for_instance_mean(self, message: Message):
        """
        The handling function for calculate instance_norm after receiving \
        var and mean.
        """
        sender = message.sender
        feat_mean = message.content
        self.global_mean = feat_mean
        stat = {}
        # Send var
        for split in ['train_data', 'val_data', 'test_data']:
            if hasattr(self.data, split):
                split_data = getattr(self.data, split)
                if split_data is not None and 'x' in split_data:
                    stat[split] = {
                        'var': np.mean(
                            (split_data['x'].T - feat_mean[split]).T**2,
                            axis=1),
                        'dim': np.shape(split_data['x'])[1],
                    }

        # Send mean
        self.comm_manager.send(
            Message(msg_type='instance_var',
                    sender=self.ID,
                    receiver=sender,
                    content=stat))

    def callback_func_for_instance_norm(self, message: Message):
        """
        The handling function for performing instance_norm after receiving \
        var and mean.
        """
        feat_var = message.content
        # Apply instance norm
        for split in ['train_data', 'val_data', 'test_data']:
            if hasattr(self.data, split):
                split_data = getattr(self.data, split)
                if split_data is not None and 'x' in split_data:
                    split_data['x'] = (
                        (split_data['x'].T - self.global_mean[split]) /
                        (feat_var[split]**0.5)).T
        logger.info('Instance norm finished.')

    # Bind method to instance
    worker.callback_func_for_ask_for_mean = types.MethodType(
        callback_func_for_ask_for_mean, worker)
    worker.callback_func_for_instance_mean = types.MethodType(
        callback_func_for_instance_mean, worker)
    worker.callback_func_for_instance_norm = types.MethodType(
        callback_func_for_instance_norm, worker)

    # Register handlers functions
    worker.register_handlers('ask_for_instance_mean',
                             worker.callback_func_for_ask_for_mean)
    worker.register_handlers('instance_mean',
                             worker.callback_func_for_instance_mean)
    worker.register_handlers('instance_var',
                             worker.callback_func_for_instance_norm)
    return worker
