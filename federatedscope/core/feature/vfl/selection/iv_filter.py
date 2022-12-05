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

        # Broadcast client address and feat_engr_public_key
        self.broadcast_client_address()
        self.comm_manager.send(
            Message(msg_type='binning',
                    sender=self.ID,
                    receiver=list(self.comm_manager.get_neighbors().keys()),
                    state=self.state,
                    content=self._cfg.feat_engr.selec_woe_binning))


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
        assert len(set(merged_y)) == 2, \
            'IV filter only available for binary classification tasks'
        if merged_feat is not None:
            num_features = merged_feat.shape[1]
            num_bins = [worker._cfg.feat_engr.num_bins] * num_features
            bin_edges = vfl_binning(merged_feat, num_bins, message.content)

        for i in range(merged_feat):
            merged_feat[:, i] = np.searchsorted(bin_edges[i][1:-1],
                                                merged_feat['x'][:, i],
                                                side="right")
        self.merged_feat = merged_feat
        self.bin_edges = bin_edges

        if self.own_label:
            # Local woe_filter
            iv_list = []
            good_t = np.sum(merged_y)
            bad_t = num_features.shape[1] - good_t
            for j in range(merged_feat.shape[1]):
                feat = merged_feat[:, j]
                iv_value = 0
                for binning in set(feat):
                    good_rate = merged_y[np.sum(
                        np.where(feat == binning)[[0]])] / good_t
                    bad_rate = (np.sum(feat == binning) - good_t * good_rate) \
                        / bad_t
                    iv_value += np.log(bad_rate/good_rate) * \
                        (bad_rate - good_rate)
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

            # Generate HE keys
            self.feat_engr_public_key, self.feat_engr_private_key = \
                secure_builder(worker._cfg).generate_keypair()

            en_y_dict = {}
            for split in ['train_data', 'val_data', 'test_data']:
                if hasattr(worker.data, split):
                    split_data = getattr(worker.data, split)
                    if split_data is not None and 'y' in split_data:
                        en_y_dict[split] = [
                            self.feat_engr_public_key.encrypt(y_i)
                            for y_i in split_data['y']
                        ]

            # Send en_y_dict to client without label
            self.comm_manager.send(
                Message(msg_type='en_y',
                        sender=self.ID,
                        receiver=[
                            each for each in self.comm_manager.neighbors
                            if each != self.server_id
                        ],
                        state=self.state,
                        content=en_y_dict))

    def callback_func_for_en_y(self, message: Message):
        en_y_dict = message.content
        sender = message.sender
