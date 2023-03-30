import types
import logging
import numpy as np

from federatedscope.core.message import Message
from federatedscope.core.feature.utils import secure_builder, \
    merge_splits_feat, vfl_binning

logger = logging.getLogger(__name__)


def wrap_iv_filter_server(worker):
    """
    This function is to perform feature selection with iv_filter \
    to data for server.
    Args:
        worker: ``federatedscope.core.workers.Worker`` to be wrapped

    Returns:
        Wrap vfl server with iv_filter.
    """
    def trigger_for_feat_engr(self,
                              trigger_train_func,
                              kwargs_for_trigger_train_func={}):
        logger.info('Start to execute woe_filter, which requires HE.')
        self.trigger_train_func = trigger_train_func
        self.kwargs_for_trigger_train_func = \
            kwargs_for_trigger_train_func
        self.msg_buffer['feat_dim'] = {}

        # Broadcast client address and feat_engr_public_key
        self.broadcast_client_address()
        self.comm_manager.send(
            Message(msg_type='binning',
                    sender=self.ID,
                    receiver=list(self.comm_manager.get_neighbors().keys()),
                    state=self.state,
                    content=self._cfg.feat_engr.selec_woe_binning))

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
                            'feat_dim'][i][1] + self._cfg.vertical.dims[i-2]))
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
    worker.callbacks_funcs_for_feat_dim = types.MethodType(
        callbacks_funcs_for_feat_dim, worker)

    # Register handlers functions
    worker.register_handlers('feat_dim', worker.callbacks_funcs_for_feat_dim)
    return worker


def wrap_iv_filter_client(worker):
    """
    This function is to perform feature selection with iv_filter \
    to data for client.
    Args:
        worker: ``federatedscope.core.workers.Worker`` to be wrapped

    Returns:
        Wrap vfl client with iv_filter.
    """
    def callback_funcs_for_binning(self, message: Message):
        # Merge train & val & test
        merged_feat, merged_y = merge_splits_feat(worker.data)

        if merged_feat is not None:
            num_features = merged_feat.shape[1]
            num_bins = [worker._cfg.feat_engr.num_bins] * num_features
            bin_edges = vfl_binning(merged_feat, num_bins, message.content)

        for i in range(merged_feat.shape[1]):
            merged_feat[:, i] = np.searchsorted(bin_edges[i][1:-1],
                                                merged_feat[:, i],
                                                side="right")
        self.merged_feat, self.merged_y = merged_feat, merged_y
        self.bin_edges = bin_edges
        self.sum_en_y_cnt = 1

        if self.own_label:
            assert len(set(merged_y)) == 2, \
                'IV filter only available for binary classification tasks'
            # Local woe_filter
            iv_list = []
            good_t = np.sum(merged_y)
            bad_t = merged_feat.shape[0] - good_t
            for j in range(merged_feat.shape[1]):
                feat = merged_feat[:, j]
                iv_value = 0
                for binning in set(feat):
                    good_rate = np.sum(
                        merged_y[np.where(feat == binning)[0]]) / good_t
                    bad_rate = (np.sum(feat == binning) - good_t * good_rate) \
                        / bad_t
                    iv_value += np.log(bad_rate/good_rate) * \
                        (bad_rate - good_rate)
                iv_list.append(iv_value)
            threshold = worker._cfg.feat_engr.selec_threshold
            filtered_col = (np.array(iv_list) < threshold).nonzero()[0]
            # Filter local feature
            for split in ['train_data', 'val_data', 'test_data']:
                if hasattr(worker.data, split):
                    split_data = getattr(worker.data, split)
                    if split_data is not None and 'x' in split_data:
                        split_data['x'] = \
                            np.delete(split_data['x'], filtered_col, axis=1)
            worker._init_data_related_var()

            self.comm_manager.send(
                Message(msg_type='feat_dim',
                        sender=self.ID,
                        receiver=[self.server_id],
                        content=(split_data['x'].shape[1], filtered_col)))

            # Generate HE keys
            self.feat_engr_public_key, self.feat_engr_private_key = \
                secure_builder(worker._cfg).generate_keypair()
            en_y = [self.feat_engr_public_key.encrypt(y_i) for y_i in merged_y]

            # Send en_y_dict to client without label
            self.comm_manager.send(
                Message(msg_type='en_y',
                        sender=self.ID,
                        receiver=[
                            each for each in self.comm_manager.neighbors
                            if each != self.server_id
                        ],
                        state=self.state,
                        content=en_y))

    def callback_func_for_en_y(self, message: Message):
        en_y = message.content
        sender = message.sender

        sum_en_y = []
        for j in range(self.merged_feat.shape[1]):
            feat = self.merged_feat[:, j]
            sum_en_y_j = []
            for binning in set(feat):
                index = np.where(feat == binning)[0]
                sum_en_y_jj = np.sum([en_y[x] for x in index])
                binning_num = len(index)
                sum_en_y_j.append((sum_en_y_jj, binning_num))
            sum_en_y.append(sum_en_y_j)

        self.comm_manager.send(
            Message(msg_type='sum_en_y',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=sum_en_y))

    def callback_func_for_sum_en_y(self, message: Message):
        sum_en_y = message.content
        sender = message.sender

        good_t = np.sum(self.merged_y)
        bad_t = len(self.merged_y) - good_t

        iv_list = []
        for j in range(len(sum_en_y)):
            sum_en_y_j = sum_en_y[j]
            iv_value = 0
            for binning in sum_en_y_j:
                sum_en_y_jj, binning_num = binning
                good_rate = self.feat_engr_private_key.decrypt(
                    sum_en_y_jj) / good_t
                bad_rate = (binning_num - good_t * good_rate) / bad_t
                iv_value += np.log(bad_rate / good_rate) * \
                    (bad_rate - good_rate)
        iv_list.append(iv_value)

        self.comm_manager.send(
            Message(msg_type='iv_list',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=iv_list))

    def callback_func_for_iv_list(self, message: Message):
        iv_list = message.content

        threshold = worker._cfg.feat_engr.selec_threshold
        filtered_col = (np.array(iv_list) < threshold).nonzero()[0]
        # Filter local feature
        for split in ['train_data', 'val_data', 'test_data']:
            if hasattr(worker.data, split):
                split_data = getattr(worker.data, split)
                if split_data is not None and 'x' in split_data:
                    split_data['x'] = \
                        np.delete(split_data['x'], filtered_col, axis=1)
        worker._init_data_related_var()

        self.comm_manager.send(
            Message(msg_type='feat_dim',
                    sender=self.ID,
                    receiver=[self.server_id],
                    content=(split_data['x'].shape[1], filtered_col)))

    def callback_funcs_for_vertical_dims(self, message: Message):
        vertical_dims = message.content
        self.vertical_dims = vertical_dims
        if hasattr(self, '_init_data_related_var'):
            self._init_data_related_var()

    # Bind method to instance
    worker.callback_funcs_for_binning = types.MethodType(
        callback_funcs_for_binning, worker)
    worker.callback_func_for_en_y = types.MethodType(callback_func_for_en_y,
                                                     worker)
    worker.callback_func_for_sum_en_y = types.MethodType(
        callback_func_for_sum_en_y, worker)
    worker.callback_func_for_iv_list = types.MethodType(
        callback_func_for_iv_list, worker)
    worker.callback_funcs_for_vertical_dims = types.MethodType(
        callback_funcs_for_vertical_dims, worker)

    # Register handlers functions
    worker.register_handlers('binning', worker.callback_funcs_for_binning)
    worker.register_handlers('en_y', worker.callback_func_for_en_y)
    worker.register_handlers('sum_en_y', worker.callback_func_for_sum_en_y)
    worker.register_handlers('iv_list', worker.callback_func_for_iv_list)
    worker.register_handlers('vertical_dims',
                             worker.callback_funcs_for_vertical_dims)

    return worker
