import numpy as np
from federatedscope.vertical_fl.trainer.trainer import VerticalTrainer


class FeatureOrderProtectedTrainer(VerticalTrainer):
    def __init__(self, model, data, device, config, monitor):
        super(FeatureOrderProtectedTrainer,
              self).__init__(model, data, device, config, monitor)

        assert config.vertical.protect_method != '', \
            "Please specify the adopted method for protecting feature order"
        args = config.vertical.protect_args[0] if len(
            config.vertical.protect_args) > 0 else {}

        if config.vertical.protect_method == 'use_bins':
            self.bin_num = args.get('bin_num', 100)
            self.share_bin = args.get('share_bin', False)
            self.protect_funcs = self._protect_via_bins
            self.split_value = None
        else:
            raise ValueError(f"The method {args['method']} is not provided")

    def get_feature_value(self, feature_idx, value_idx):
        assert self.split_value is not None

        return self.split_value[feature_idx][value_idx]

    def _protect_via_bins(self, raw_feature_order, data):
        protected_feature_order = list()
        bin_size = int(np.ceil(self.cfg.dataloader.batch_size / self.bin_num))
        split_position = [[] for _ in range(len(raw_feature_order))
                          ] if self.share_bin else None
        self.split_value = [dict() for _ in range(len(raw_feature_order))]
        for i in range(len(raw_feature_order)):
            _protected_feature_order = list()
            for j in range(self.bin_num):
                idx_start = j * bin_size
                idx_end = min((j + 1) * bin_size, len(raw_feature_order[i]))
                feature_order_frame = raw_feature_order[i][idx_start:idx_end]
                np.random.shuffle(feature_order_frame)
                _protected_feature_order.append(feature_order_frame)
                if self.share_bin:
                    if j != self.bin_num - 1:
                        split_position[i].append(idx_end)
                    min_value = min(data[feature_order_frame][:, i])
                    max_value = max(data[feature_order_frame][:, i])
                    if j == 0:
                        self.split_value[i][idx_end] = max_value
                    elif j == self.bin_num - 1:
                        self.split_value[i][idx_start] += min_value / 2.0
                    else:
                        self.split_value[i][idx_start] += min_value / 2.0
                        self.split_value[i][idx_end] = max_value / 2.0
                else:
                    mean_value = np.mean(data[feature_order_frame][:, i])
                    for x in range(idx_start, idx_end):
                        self.split_value[i][x] = mean_value
            protected_feature_order.append(
                np.concatenate(_protected_feature_order))

        extra_info = None
        if split_position is not None:
            extra_info = {'split_position': split_position}

        return {
            'feature_order': protected_feature_order,
            'extra_info': extra_info
        }

    def _get_feature_order_info(self, data):
        num_of_feature = data.shape[1]
        feature_order = [0] * num_of_feature
        for i in range(num_of_feature):
            feature_order[i] = data[:, i].argsort()
        return self.protect_funcs(feature_order, data)
