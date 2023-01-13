import types
import logging
import numpy as np

from federatedscope.core.message import Message
from federatedscope.core.secret_sharing import AdditiveSecretSharing

logger = logging.getLogger(__name__)


def wrap_instance_norm_server(worker):
    """
    This function is to perform instance norm on vfl tabular data for server.
    Args:
        worker: ``federatedscope.core.workers.Worker`` to be wrapped

    Returns:
        Wrap vfl server with instance norm.
    """
    def trigger_for_feat_engr(self,
                              trigger_train_func,
                              kwargs_for_trigger_train_func={}):
        # broadcast_model_para_func after feature engineering finishing
        self.trigger_train_func = trigger_train_func
        self.kwargs_for_trigger_train_func = \
            kwargs_for_trigger_train_func

        logger.info('Start to execute instance norm with secret sharing.')

        # Broadcast client address for ss
        self.broadcast_client_address()
        self.ss_manager = AdditiveSecretSharing(
            shared_party_num=int(self._cfg.federate.client_num))

        # Ask for instance statistics
        self.msg_buffer['ss_instance_sum'] = []
        self.msg_buffer['ss_instance_sum_norm_square'] = []
        self.comm_manager.send(
            Message(msg_type='ask_for_instance_sum',
                    sender=self.ID,
                    receiver=list(self.comm_manager.neighbors.keys()),
                    timestamp=self.cur_timestamp))

    def callback_func_for_ss_instance_sum(self, message: Message):
        """
        The handling function for aggregating ss_instance_sum from all clients.
        """
        content = message.content
        self.msg_buffer['ss_instance_sum'].append(content)

        if len(self.msg_buffer['ss_instance_sum']) == self._client_num:
            logger.info('Sever secret reconstruct ss_instance_sum.')
            # Aggregate ss_instance_sum
            instance_mean = {}
            for split in self.msg_buffer['ss_instance_sum'][0]:
                instance_mean[split] = self.ss_manager.secret_reconstruct([
                    content[split]
                    for content in self.msg_buffer['ss_instance_sum']
                ])
                instance_mean[split] = instance_mean[split][
                    'sum'] / self._cfg.vertical.dims[-1]
            logger.info('Server send instance_mean to clients')
            self.global_mean = instance_mean
            self.comm_manager.send(
                Message(msg_type='instance_mean',
                        sender=self.ID,
                        receiver=list(self.comm_manager.neighbors.keys()),
                        timestamp=self.cur_timestamp,
                        content=instance_mean))

    def callback_func_for_ss_instance_sum_norm_square(self, message: Message):
        """
        The handling function for aggregating instance_var from all clients.
        """
        content = message.content
        self.msg_buffer['ss_instance_sum_norm_square'].append(content)

        if len(self.msg_buffer['ss_instance_sum_norm_square']) == \
                self._client_num:
            logger.info('Sever secret reconstruct '
                        'ss_instance_sum_norm_square.')
            # Aggregate ss_instance_sum_norm_square
            instance_var = {}
            for split in self.msg_buffer['ss_instance_sum'][0]:
                instance_var[split] = self.ss_manager.secret_reconstruct([
                    content[split] for content in
                    self.msg_buffer['ss_instance_sum_norm_square']
                ])
                instance_var[split] = instance_var[split][
                    'sum_norm_square'] / self._cfg.vertical.dims[-1]

            # Apply instance norm
            for split in ['train_data', 'val_data', 'test_data']:
                if hasattr(self.data, split):
                    split_data = getattr(self.data, split)
                    if split_data is not None and 'x' in split_data:
                        split_data['x'] = (
                            (split_data['x'].T - self.global_mean[split]) /
                            (instance_var[split]**0.5)).T

            self.comm_manager.send(
                Message(msg_type='instance_var',
                        sender=self.ID,
                        receiver=list(self.comm_manager.neighbors.keys()),
                        timestamp=self.cur_timestamp,
                        content=instance_var))
            self._init_data_related_var()
            self.trigger_train_func(**self.kwargs_for_trigger_train_func)

    # Bind method to instance
    worker.trigger_for_feat_engr = types.MethodType(trigger_for_feat_engr,
                                                    worker)
    worker.callback_func_for_ss_instance_sum = types.MethodType(
        callback_func_for_ss_instance_sum, worker)
    worker.callback_func_for_ss_instance_sum_norm_square = types.MethodType(
        callback_func_for_ss_instance_sum_norm_square, worker)

    # Register handlers functions
    worker.register_handlers('ss_instance_sum',
                             worker.callback_func_for_ss_instance_sum)
    worker.register_handlers(
        'ss_instance_sum_norm_square',
        worker.callback_func_for_ss_instance_sum_norm_square)
    return worker


def wrap_instance_norm_client(worker):
    """
    This function is to perform instance norm vfl tabular data for client.
    Args:
        worker: ``federatedscope.core.workers.Worker`` to be wrapped

    Returns:
        Wrap vfl client with instance norm.
    """
    def callback_func_for_ask_for_instance_sum(self, message: Message):
        self.ss_manager = AdditiveSecretSharing(
            shared_party_num=int(self._cfg.federate.client_num))
        self.msg_buffer['ss_instance_sum'] = {}
        content = {}
        # Calculate sum
        for split in ['train_data', 'val_data', 'test_data']:
            if hasattr(self.data, split):
                split_data = getattr(self.data, split)
                if split_data is not None and 'x' in split_data:
                    content[split] = self.ss_manager.secret_split(
                        {'sum': np.sum(split_data['x'], axis=1)})
        # Self-hosted ss_instance_sum
        self.msg_buffer['ss_instance_sum'][self.ID] = {
            key: value[self.ID - 1]
            for key, value in content.items()
        }

        # Send ss split to neighbors
        for neighbor in self.comm_manager.neighbors:
            content_frame = {
                key: value[neighbor - 1]
                for key, value in content.items()
            }

            self.comm_manager.send(
                Message(msg_type='ss_instance_sum',
                        sender=self.ID,
                        receiver=[neighbor],
                        content=content_frame))

    def callback_func_ss_instance_sum(self, message: Message):
        sender = message.sender
        content_frame = message.content
        self.msg_buffer['ss_instance_sum'][sender] = content_frame
        logger.info(f'Client {self.ID} receive Client {sender} '
                    f'ss_instance_sum')

        if len(self.msg_buffer['ss_instance_sum'].keys()) == \
                self._cfg.federate.client_num:
            # Sum up ss_instance_sum
            content = {}
            for sender in self.msg_buffer['ss_instance_sum'].keys():
                content_frame = self.msg_buffer['ss_instance_sum'][sender]
                for key, value in content_frame.items():
                    if key not in content:
                        content[key] = value
                    else:
                        for sub_key in content[key].keys():
                            content[key][sub_key] += content_frame[key][
                                sub_key]

            self.comm_manager.send(
                Message(msg_type='ss_instance_sum',
                        sender=self.ID,
                        receiver=[self.server_id],
                        content=content))

    def callback_func_for_instance_mean(self, message: Message):
        """
        The handling function for calculate instance_norm after receiving \
        var and mean.
        """
        sender = message.sender
        feat_mean = message.content
        self.global_mean = feat_mean

        self.msg_buffer['ss_instance_sum_norm_square'] = {}  # For variance
        content = {}
        # Send norm_square
        for split in ['train_data', 'val_data', 'test_data']:
            if hasattr(self.data, split):
                split_data = getattr(self.data, split)
                if split_data is not None and 'x' in split_data:
                    content[split] = self.ss_manager.secret_split({
                        'sum_norm_square': np.sum(
                            (split_data['x'].T - feat_mean[split]).T**2,
                            axis=1)
                    })

        # Self-hosted ss_instance_sum_norm_square
        self.msg_buffer['ss_instance_sum_norm_square'][self.ID] = {
            key: value[self.ID - 1]
            for key, value in content.items()
        }

        # Send ss split to neighbors
        for neighbor in self.comm_manager.neighbors:
            content_frame = {
                key: value[neighbor - 1]
                for key, value in content.items()
            }

            self.comm_manager.send(
                Message(msg_type='ss_instance_sum_norm_square',
                        sender=self.ID,
                        receiver=[neighbor],
                        content=content_frame))

    def callback_func_for_ss_instance_sum_norm_square(self, message: Message):
        sender = message.sender
        content_frame = message.content
        self.msg_buffer['ss_instance_sum_norm_square'][sender] = content_frame
        logger.info(f'Client {self.ID} receive Client {sender} '
                    f'ss_instance_sum_norm_square')

        if len(self.msg_buffer['ss_instance_sum_norm_square'].keys()) == \
                self._cfg.federate.client_num:
            # Sum up ss_instance_sum_norm_square
            content = {}
            for sender in self.msg_buffer['ss_instance_sum_norm_square'].keys(
            ):
                content_frame = self.msg_buffer['ss_instance_sum_norm_square'][
                    sender]
                for key, value in content_frame.items():
                    if key not in content:
                        content[key] = value
                    else:
                        for sub_key in content[key].keys():
                            content[key][sub_key] += content_frame[key][
                                sub_key]

            self.comm_manager.send(
                Message(msg_type='ss_instance_sum_norm_square',
                        sender=self.ID,
                        receiver=[self.server_id],
                        content=content))

    def callback_func_for_instance_var(self, message: Message):
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
        self.msg_buffer.pop('ss_instance_sum_norm_square')
        self.msg_buffer.pop('ss_instance_sum')
        self._init_data_related_var()

    # Bind method to instance
    worker.callback_func_for_ask_for_instance_sum = types.MethodType(
        callback_func_for_ask_for_instance_sum, worker)
    worker.callback_func_ss_instance_sum = types.MethodType(
        callback_func_ss_instance_sum, worker)
    worker.callback_func_for_instance_mean = types.MethodType(
        callback_func_for_instance_mean, worker)
    worker.callback_func_for_ss_instance_sum_norm_square = types.MethodType(
        callback_func_for_ss_instance_sum_norm_square, worker)
    worker.callback_func_for_instance_var = types.MethodType(
        callback_func_for_instance_var, worker)

    # Register handlers functions
    worker.register_handlers('ask_for_instance_sum',
                             worker.callback_func_for_ask_for_instance_sum)
    worker.register_handlers('ss_instance_sum',
                             worker.callback_func_ss_instance_sum)
    worker.register_handlers('instance_mean',
                             worker.callback_func_for_instance_mean)
    worker.register_handlers(
        'ss_instance_sum_norm_square',
        worker.callback_func_for_ss_instance_sum_norm_square)
    worker.register_handlers('instance_var',
                             worker.callback_func_for_instance_var)
    return worker
