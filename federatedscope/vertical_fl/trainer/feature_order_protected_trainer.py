import numpy as np
from federatedscope.vertical_fl.trainer.utils import bucketize


def createFeatureOrderProtectedTrainer(cls, model, data, device, config,
                                       monitor):
    class FeatureOrderProtectedTrainer(cls):
        def __init__(self, model, data, device, config, monitor):
            super(FeatureOrderProtectedTrainer,
                  self).__init__(model, data, device, config, monitor)

            assert config.vertical.protect_method != '', \
                "Please specify the method for protecting feature order"
            args = config.vertical.protect_args[0] if len(
                config.vertical.protect_args) > 0 else {}

            if config.vertical.protect_method == 'dp':
                self.bucket_num = args.get('bucket_num', 100)
                self.epsilon = args.get('epsilon', None)
                self.protect_funcs = self._protect_via_dp
                self.split_value = None
            elif config.vertical.protect_method == 'op_boost':
                self.algo = args.get('algo', 'global')
                self.protect_funcs = self._protect_via_op_boost
                self.lower_bound = args.get('lower_bound', 1)
                self.upper_bound = args.get('upper_bound', 100)
                self.bucket_num = args.get('bucket_num', None)
                if self.algo == 'global':
                    self.epsilon = args.get('epsilon', 2)
                elif self.algo == 'adjusting':
                    self.epsilon_prt = args.get('epsilon_prt', 2)
                    self.epsilon_ner = args.get('epsilon_ner', 2)
                    self.partition_num = args.get('partition_num', 10)
                else:
                    raise ValueError
            else:
                raise ValueError(
                    f"The method {args['method']} is not provided")

        def get_feature_value(self, feature_idx, value_idx):
            if not hasattr(self, 'split_value') or self.split_value is None:
                return super().get_feature_value(feature_idx=feature_idx,
                                                 value_idx=value_idx)

            return self.split_value[feature_idx][value_idx]

        def _processed_data(self, data):
            min_value = np.min(data, axis=0)
            max_value = np.max(data, axis=0)
            # To avoid data_max[i] == data_min[i],
            for i in range(data.shape[1]):
                if max_value[i] == min_value[i]:
                    max_value[i] += 1
            return np.round(self.lower_bound + (data - min_value) /
                            (max_value - min_value) *
                            (self.upper_bound - self.lower_bound))

        def _global_mapping_fun(self, x, epsilon, lower_bound, upper_bound):
            probs = list()
            denominator = np.sum(
                np.exp(
                    -np.abs(x - np.array(range(lower_bound, upper_bound + 1)))
                    * epsilon / 2))
            for k in range(lower_bound, upper_bound + 1):
                probs.append(
                    np.exp(-np.abs(x - k) * epsilon / 2) / denominator)
            res = np.random.choice(list(range(lower_bound, upper_bound + 1)),
                                   p=probs)

            return res

        def _adjusting_mapping_fun(self, x, partition_edges):
            for part_idx in range(self.partition_num):
                if partition_edges[part_idx] < x <= partition_edges[part_idx +
                                                                    1]:
                    selected_part = self._global_mapping_fun(
                        part_idx,
                        epsilon=self.epsilon_prt,
                        lower_bound=0,
                        upper_bound=self.partition_num - 1)
                    res = self._global_mapping_fun(
                        x,
                        epsilon=self.epsilon_ner,
                        lower_bound=partition_edges[selected_part] + 1,
                        upper_bound=partition_edges[selected_part + 1])

                    return res

        def _op_boost_global(self, data):

            processed_data = self._processed_data(data=data)
            mapped_data = np.vectorize(self._global_mapping_fun)(
                processed_data,
                epsilon=self.epsilon,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound)

            return mapped_data

        def _op_boost_adjusting(self, data):

            processed_data = self._processed_data(data=data)
            quantiles = np.linspace(0, 100, self.partition_num + 1)
            partition_edges = np.round(
                np.asarray(
                    np.percentile(
                        list(range(self.lower_bound - 1,
                                   self.upper_bound + 1)), quantiles)))
            partition_edges = [int(x) for x in partition_edges]
            mapped_data = np.vectorize(self._adjusting_mapping_fun,
                                       signature='(),(n)->()')(
                                           processed_data,
                                           partition_edges=partition_edges)

            return mapped_data

        def _protect_via_op_boost(self, raw_feature_order, data):
            """
                Add random noises to feature order for privacy protection.
                For more details, please see
                OpBoost: A Vertical Federated Tree Boosting Framework Based on
                    Order-Preserving Desensitization.pdf
                (https://arxiv.org/pdf/2210.01318.pdf)
            """
            if self.algo == 'global':
                mapped_data = self._op_boost_global(data)
            elif self.algo == 'adjusting':
                mapped_data = self._op_boost_adjusting(data)
            else:
                mapped_data = None
            assert mapped_data is not None

            # Get feature order based on mapped data
            num_of_feature = mapped_data.shape[1]
            protected_feature_order = [0] * num_of_feature
            for i in range(num_of_feature):
                protected_feature_order[i] = mapped_data[:, i].argsort()

            # add bucket in op_boost to accelerate
            if self.bucket_num:
                new_protected_feature_order = list()
                bucket_size = int(np.floor(data.shape[0] / self.bucket_num))
                split_position = []
                self.split_value = []

                for feature_idx in range(len(raw_feature_order)):
                    bucketized_feature_order = bucketize(
                        raw_feature_order[feature_idx], bucket_size,
                        self.bucket_num)
                    bucketized_protected_feature_order = bucketize(
                        protected_feature_order[feature_idx], bucket_size,
                        self.bucket_num)

                    split_position = self.compute_extra_info(
                        data, split_position, feature_idx,
                        bucketized_feature_order, bucketized_feature_order)

                    bucketized_protected_feature_order = [
                        x for x in bucketized_protected_feature_order
                        if len(x) > 0
                    ]

                    bucketized_protected_feature_order = np.concatenate(
                        bucketized_protected_feature_order)
                    new_protected_feature_order.append(
                        bucketized_protected_feature_order)

                extra_info = {'split_position': split_position}

                return {
                    'raw_feature_order': raw_feature_order,
                    'feature_order': new_protected_feature_order,
                    'extra_info': extra_info
                }
            else:
                return {
                    'raw_feature_order': raw_feature_order,
                    'feature_order': protected_feature_order,
                }

        def _protect_via_dp(self, raw_feature_order, data):
            """
                Bucketize and add dp noise to feature order for protection.
                For more details, please refer to
                    FederBoost: Private Federated Learning for GBDT
                    (https://arxiv.org/pdf/2011.02796.pdf)
            """
            protected_feature_order = list()
            bucket_size = int(np.floor(data.shape[0] / self.bucket_num))
            if self.epsilon is None:
                prob_for_preserving = 1.0
            else:
                _tmp = np.power(np.e, self.epsilon)
                prob_for_preserving = _tmp / (_tmp + self.bucket_num - 1)
            prob_for_moving = (1.0 - prob_for_preserving) / (self.bucket_num -
                                                             1)
            split_position = []
            self.split_value = []

            for feature_idx in range(len(raw_feature_order)):
                bucketized_feature_order = bucketize(
                    raw_feature_order[feature_idx], bucket_size,
                    self.bucket_num)
                noisy_bucketized_feature_order = [
                    [] for _ in range(self.bucket_num)
                ]

                # Add noise to bucketized feature order
                for bucket_idx in range(self.bucket_num):
                    probs = np.ones(self.bucket_num) * prob_for_moving
                    probs[bucket_idx] = prob_for_preserving
                    for each in bucketized_feature_order[bucket_idx]:
                        selected_bucket_idx = np.random.choice(list(
                            range(self.bucket_num)),
                                                               p=probs)
                        noisy_bucketized_feature_order[
                            selected_bucket_idx].append(each)

                # Save split positions (instance number within buckets)
                # We exclude the endpoints to avoid empty sub-trees
                split_position = self.compute_extra_info(
                    data, split_position, feature_idx,
                    noisy_bucketized_feature_order, bucketized_feature_order)

                noisy_bucketized_feature_order = [
                    x for x in noisy_bucketized_feature_order if len(x) > 0
                ]
                [np.random.shuffle(x) for x in noisy_bucketized_feature_order]
                noisy_bucketized_feature_order = np.concatenate(
                    noisy_bucketized_feature_order)
                protected_feature_order.append(noisy_bucketized_feature_order)

            extra_info = {'split_position': split_position}

            return {
                'raw_feature_order': raw_feature_order,
                'feature_order': protected_feature_order,
                'extra_info': extra_info
            }

        def compute_extra_info(self, data, split_position, feature_idx,
                               noisy_bucketized_feature_order,
                               bucketized_feature_order):
            # Save split positions (instance number within buckets)
            # We exclude the endpoints to avoid empty sub-trees
            _split_position = list()
            _split_value = dict()
            accumu_num = 0
            for bucket_idx, each_bucket in enumerate(
                    noisy_bucketized_feature_order):
                instance_num = len(each_bucket)
                # Skip the empty bucket
                if instance_num != 0:
                    # Skip the endpoints
                    if bucket_idx != self.bucket_num - 1:
                        _split_position.append(accumu_num + instance_num)

                    # Save split values: average of min value of (j-1)-th
                    # bucket and max value of j-th bucket
                    max_value = data[bucketized_feature_order[bucket_idx]
                                     [-1]][feature_idx]
                    min_value = data[bucketized_feature_order[bucket_idx]
                                     [0]][feature_idx]
                    if accumu_num == 0:
                        _split_value[accumu_num +
                                     instance_num] = max_value / 2.0
                    elif bucket_idx == self.bucket_num - 1:
                        _split_value[accumu_num] += min_value / 2.0
                    else:
                        _split_value[accumu_num] += min_value / 2.0
                        _split_value[accumu_num +
                                     instance_num] = max_value / 2.0

                    accumu_num += instance_num

            split_position.append(_split_position)
            self.split_value.append(_split_value)
            return split_position

        # TODO: more flexible for client to choose whether to protect or not
        def _get_feature_order_info(self, data):
            num_of_feature = data.shape[1]
            feature_order = [0] * num_of_feature
            for i in range(num_of_feature):
                feature_order[i] = data[:, i].argsort()
            return self.protect_funcs(feature_order, data)

    return FeatureOrderProtectedTrainer(model, data, device, config, monitor)
