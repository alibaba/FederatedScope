import types
import logging
import numpy as np

from federatedscope.core.message import Message
from federatedscope.core.feature.utils import secure_builder, merge_splits_feat

logger = logging.getLogger(__name__)


def wrap_correlation_filter_server(worker):
    """
    This function is to perform feature selection with correlation_filter \
    to data for server.
    Args:
        worker: ``federatedscope.core.workers.Worker`` to be wrapped

    Returns:
        Wrap vfl server with correlation_filter.
    """
    def trigger_for_feat_engr(self,
                              trigger_train_func,
                              kwargs_for_trigger_train_func={}):
        logger.info('Start to execute correlation_filter, which requires FHE.')

        self.msg_buffer['feat_dim'] = {}
        # broadcast_model_para_func after feature engineering finishing
        self.trigger_train_func = trigger_train_func
        self.kwargs_for_trigger_train_func = \
            kwargs_for_trigger_train_func

        # Broadcast client address and feat_engr_public_key
        self.broadcast_client_address()
        self.feat_engr_public_key, self.feat_engr_private_key = \
            secure_builder(worker._cfg).generate_keypair()
        logger.info('Sending feat_engr_public_keys to clients.')
        self.comm_manager.send(
            Message(msg_type='feat_engr_public_keys',
                    sender=self.ID,
                    receiver=list(self.comm_manager.get_neighbors().keys()),
                    state=self.state,
                    content=self.feat_engr_public_key))

    def callback_funcs_for_en_feat_corrcoef(self, message: Message):
        receiver, en_feat_corrcoef = message.content

        feat_corrcoef = [
            self.feat_engr_private_key.decrypt(x) for x in en_feat_corrcoef
        ]
        logger.info('Sending correlation coefficient to clients.')
        self.comm_manager.send(
            Message(msg_type='feat_corrcoef',
                    sender=self.ID,
                    receiver=receiver,
                    state=self.state,
                    content=feat_corrcoef))

    def callbacks_funcs_for_feat_dim(self, message: Message):
        feat_dim, filtered_col = message.content
        sender = message.sender

        self.msg_buffer['feat_dim'][sender] = (feat_dim, filtered_col)
        if len(self.msg_buffer['feat_dim']) == self._client_num:
            server_filtered_col = None
            vertical_dims = [0]
            for i in sorted(list(self.msg_buffer['feat_dim'].keys())):
                vertical_dims.append(
                    sum(vertical_dims) + self.msg_buffer['feat_dim'][i][0])
                if server_filtered_col is None:
                    server_filtered_col = self.msg_buffer['feat_dim'][i][1]
                else:
                    server_filtered_col = \
                        np.hstack((server_filtered_col, self.msg_buffer[
                            'feat_dim'][i][1]+self.vertical_dims[i-2]))
            # Filter feature
            for split in ['train_data', 'val_data', 'test_data']:
                if hasattr(worker.data, split):
                    split_data = getattr(worker.data, split)
                    if split_data is not None and 'x' in split_data:
                        split_data['x'] = \
                            np.delete(split_data['x'], server_filtered_col,
                                      axis=1)

            vertical_dims.pop(0)
            self.comm_manager.send(
                Message(msg_type='vertical_dims',
                        sender=self.ID,
                        receiver=list(
                            self.comm_manager.get_neighbors().keys()),
                        state=self.state,
                        content=vertical_dims))
            self.vertical_dims = vertical_dims
            if hasattr(self, '_init_data_related_var'):
                self._init_data_related_var()
            self.msg_buffer.pop('feat_dim')
            self.trigger_train_func(**self.kwargs_for_trigger_train_func)

    # Bind method to instance
    worker.trigger_for_feat_engr = types.MethodType(trigger_for_feat_engr,
                                                    worker)
    worker.callback_funcs_for_en_feat_corrcoef = types.MethodType(
        callback_funcs_for_en_feat_corrcoef, worker)
    worker.callbacks_funcs_for_feat_dim = types.MethodType(
        callbacks_funcs_for_feat_dim, worker)

    # Register handlers functions
    worker.register_handlers('en_feat_corrcoef',
                             worker.callback_funcs_for_en_feat_corrcoef)
    worker.register_handlers('feat_dim', worker.callbacks_funcs_for_feat_dim)
    return worker


def wrap_correlation_filter_client(worker):
    """
    This function is to perform feature selection with correlation_filter \
    to data for client.
    Args:
        worker: ``federatedscope.core.workers.Worker`` to be wrapped

    Returns:
        Wrap vfl client with correlation_filter.
    """
    def callback_funcs_for_feat_engr_public_keys(self, message: Message):
        self.feat_engr_public_key = message.content
        if self.own_label:
            self.msg_buffer['encrypted_norm_feat'] = {}
            logger.info(f'Client {self.ID} ask_for_encrypted_norm_feat.')
            self.comm_manager.send(
                Message(msg_type='ask_for_encrypted_norm_feat',
                        sender=self.ID,
                        receiver=[
                            each for each in self.comm_manager.neighbors
                            if each != self.server_id
                        ],
                        state=self.state,
                        content=None))

    def callback_funcs_for_ask_for_encrypted_norm_feat(self, message: Message):
        sender = message.sender
        merged_feat, _ = merge_splits_feat(worker.data)
        norm_feat = (merged_feat - np.mean(merged_feat, axis=0)) / (
            np.std(merged_feat, axis=0) * merged_feat.shape[0])

        en_norm_feat = [[self.feat_engr_public_key.encrypt(j) for j in i]
                        for i in norm_feat]
        logger.info(
            f'Client {self.ID} sending encrypted_norm_feat to {sender}.')
        self.comm_manager.send(
            Message(msg_type='encrypted_norm_feat',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=en_norm_feat))

    def callback_funcs_for_encrypted_norm_feat(self, message: Message):
        if not self.own_label:
            raise NotImplementedError(f'Client {self.ID} do not have y.')

        en_norm_feat = message.content
        sender = message.sender
        self.msg_buffer['encrypted_norm_feat'][sender] = en_norm_feat

        if len(self.msg_buffer['encrypted_norm_feat'].keys()) == \
                len([each for each in self.comm_manager.neighbors if each !=
                    self.server_id]):
            threshold = worker._cfg.feat_engr.selec_threshold
            merged_feat, merged_y = merge_splits_feat(worker.data)

            # Filter local feature
            if merged_feat is not None:
                feat_corrcoef = []
                for i in range(merged_feat.shape[1]):
                    feat_corrcoef.append(
                        np.sum(
                            (merged_feat[:, i] - np.mean(merged_feat[:, i])) *
                            (merged_y - np.mean(merged_y)) /
                            merged_feat.shape[0] /
                            (np.std(merged_feat[:, i]) * np.std(merged_y))))
                filtered_col = (np.array(feat_corrcoef) <
                                threshold).nonzero()[0]
                logger.info(f'The eliminated feature of Client {self.ID} is'
                            f' {filtered_col}')
                # Filter feature
                for split in ['train_data', 'val_data', 'test_data']:
                    if hasattr(worker.data, split):
                        split_data = getattr(worker.data, split)
                        if split_data is not None and 'x' in split_data:
                            split_data['x'] = \
                                np.delete(split_data['x'], filtered_col,
                                          axis=1)
            self.comm_manager.send(
                Message(msg_type='feat_dim',
                        sender=self.ID,
                        receiver=[self.server_id],
                        content=(split_data['x'].shape[1], filtered_col)))

            norm_y = (merged_y - np.mean(merged_y)) / np.std(merged_y)
            # Calculate correlation coefficient
            for sender, en_norm_feat in \
                    self.msg_buffer['encrypted_norm_feat'].items():
                en_feat_corrcoef = []
                for i in range(np.array(en_norm_feat).shape[1]):
                    en_feat_corrcoef.append(
                        np.sum(np.array(en_norm_feat)[:, i] * norm_y))

                # Send to server for decryption
                logger.info(f'Client {self.ID} send en_feat_corrcoef to'
                            f' {self.server_id}.')
                self.comm_manager.send(
                    Message(msg_type='en_feat_corrcoef',
                            sender=self.ID,
                            receiver=[self.server_id],
                            content=(sender, en_feat_corrcoef)))

    def callbacks_funcs_for_feat_corrcoef(self, message: Message):
        feat_corrcoef = message.content

        threshold = worker._cfg.feat_engr.selec_threshold
        filtered_col = (np.array(feat_corrcoef) < threshold).nonzero()[0]
        logger.info(f'The eliminated feature of Client {self.ID} is'
                    f' {filtered_col}')

        # Filter feature
        for split in ['train_data', 'val_data', 'test_data']:
            if hasattr(worker.data, split):
                split_data = getattr(worker.data, split)
                if split_data is not None and 'x' in split_data:
                    split_data['x'] = \
                        np.delete(split_data['x'], filtered_col, axis=1)

        self.comm_manager.send(
            Message(msg_type='feat_dim',
                    sender=self.ID,
                    receiver=[self.server_id],
                    content=(split_data['x'].shape[1], filtered_col)))

    def callback_funcs_for_vertical_dims(self, message: Message):
        vertical_dims = message.content
        self.vertical_dims = vertical_dims
        if self.own_label:
            self.msg_buffer.pop('encrypted_norm_feat')
        if hasattr(self, '_init_data_related_var'):
            self._init_data_related_var()

    # Bind method to instance
    worker.callback_funcs_for_feat_engr_public_keys = types.MethodType(
        callback_funcs_for_feat_engr_public_keys, worker)
    worker.callback_funcs_for_ask_for_encrypted_norm_feat = types.MethodType(
        callback_funcs_for_ask_for_encrypted_norm_feat, worker)
    worker.callback_funcs_for_encrypted_norm_feat = types.MethodType(
        callback_funcs_for_encrypted_norm_feat, worker)
    worker.callbacks_funcs_for_feat_corrcoef = types.MethodType(
        callbacks_funcs_for_feat_corrcoef, worker)
    worker.callback_funcs_for_vertical_dims = types.MethodType(
        callback_funcs_for_vertical_dims, worker)

    # Register handlers functions
    worker.register_handlers('feat_engr_public_keys',
                             worker.callback_funcs_for_feat_engr_public_keys)
    worker.register_handlers(
        'ask_for_encrypted_norm_feat',
        worker.callback_funcs_for_ask_for_encrypted_norm_feat)
    worker.register_handlers('encrypted_norm_feat',
                             worker.callback_funcs_for_encrypted_norm_feat)
    worker.register_handlers('feat_corrcoef',
                             worker.callbacks_funcs_for_feat_corrcoef)
    worker.register_handlers('vertical_dims',
                             worker.callback_funcs_for_vertical_dims)

    return worker
