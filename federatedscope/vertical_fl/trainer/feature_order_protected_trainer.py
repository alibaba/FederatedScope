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

        if config.vertical.protect_method == 'dp':
            self.bucket_num = args.get('bucket_num', 100)
            self.epsilon = args.get('epsilon', None)
            self.protect_funcs = self._protect_via_dp
            self.split_value = None
        else:
            raise ValueError(f"The method {args['method']} is not provided")

    def get_feature_value(self, feature_idx, value_idx):
        assert self.split_value is not None

        return self.split_value[feature_idx][value_idx]

    def _bucketize(self, feature_order, bucket_size, bucket_num):
        bucketized_feature_order = list()
        for bucket_idx in range(bucket_num):
            start = bucket_idx * bucket_size
            end = min((bucket_idx + 1) * bucket_size, len(feature_order))
            bucketized_feature_order.append(feature_order[start:end])
        return bucketized_feature_order

    def _protect_via_dp(self, raw_feature_order, data):
        protected_feature_order = list()
        bucket_size = int(
            np.ceil(self.cfg.dataloader.batch_size / self.bucket_num))
        if self.epsilon is None:
            prob_for_preserving = 1.0
        else:
            _tmp = np.power(np.e, self.epsilon)
            prob_for_preserving = _tmp / (_tmp + self.bucket_num - 1)
        prob_for_moving = (1.0 - prob_for_preserving) / (self.bucket_num - 1)
        split_position = []
        self.split_value = []

        for feature_idx in range(len(raw_feature_order)):
            bucketized_feature_order = self._bucketize(
                raw_feature_order[feature_idx], bucket_size, self.bucket_num)
            noisy_bucketizd_feature_order = [[]
                                             for _ in range(self.bucket_num)]

            # Add noise to bucketized feature order
            for bucket_idx in range(self.bucket_num):
                probs = np.ones(self.bucket_num) * prob_for_moving
                probs[bucket_idx] = prob_for_preserving
                for each in bucketized_feature_order[bucket_idx]:
                    selected_bucket_idx = np.random.choice(list(
                        range(self.bucket_num)),
                                                           p=probs)
                    noisy_bucketizd_feature_order[selected_bucket_idx].append(
                        each)

            # Save split positions (instance number within buckets)
            # We exclude the endpoints to avoid empty sub-trees
            _split_position = list()
            _split_value = dict()
            accumu_num = 0
            for bucket_idx, each_bucket in enumerate(
                    noisy_bucketizd_feature_order):
                instance_num = len(each_bucket)
                # Skip the empty bucket
                if instance_num != 0:
                    # Skip the endpoints
                    if bucket_idx != self.bucket_num - 1:
                        _split_position.append(accumu_num + instance_num)

                    # Save split values: average of min value of (j-1)-th
                    # bucket and max value of j-th bucket
                    max_value = data[bucketized_feature_order[bucket_idx]
                                     [0]][feature_idx]
                    min_value = data[bucketized_feature_order[bucket_idx]
                                     [-1]][feature_idx]
                    if accumu_num == 0:
                        _split_value[accumu_num +
                                     instance_num] = min_value / 2.0
                    elif bucket_idx == self.bucket_num - 1:
                        _split_value[accumu_num] += max_value / 2.0
                    else:
                        _split_value[accumu_num] += max_value / 2.0
                        _split_value[accumu_num +
                                     instance_num] = min_value / 2.0

                    accumu_num += instance_num

            split_position.append(_split_position)
            self.split_value.append(_split_value)

            [np.random.shuffle(x) for x in noisy_bucketizd_feature_order]
            noisy_bucketizd_feature_order = np.concatenate(
                noisy_bucketizd_feature_order)
            protected_feature_order.append(noisy_bucketizd_feature_order)

        extra_info = {'split_position': split_position}

        return {
            'raw_feature_order': raw_feature_order,
            'feature_order': protected_feature_order,
            'extra_info': extra_info
        }

    def _get_feature_order_info(self, data):
        num_of_feature = data.shape[1]
        feature_order = [0] * num_of_feature
        for i in range(num_of_feature):
            feature_order[i] = data[:, i].argsort()
        return self.protect_funcs(feature_order, data)
