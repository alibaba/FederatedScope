import numpy as np
from federatedscope.vertical_fl.trainer.utils import bucketize


def createLabelProtectedTrainer(cls, model, data, device, config, monitor):
    class LabelProtectedTrainer(cls):
        def __init__(self, model, data, device, config, monitor):
            super(LabelProtectedTrainer,
                  self).__init__(model, data, device, config, monitor)

            assert config.vertical.protect_method != '', \
                "Please specify the method for protecting label"
            args = config.vertical.protect_args[0] if len(
                config.vertical.protect_args) > 0 else {}

            if config.vertical.protect_method == 'he':
                self.bucket_num = args.get('bucket_num', 100)
                self.split_value = None
                from federatedscope.vertical_fl.Paillier import \
                    abstract_paillier
                keys = abstract_paillier.generate_paillier_keypair(
                    n_length=self.cfg.vertical.key_size)
                self.public_key, self.private_key = keys
            else:
                raise ValueError(
                    f"The method {args['method']} is not provided")

        def _bucketize(self, raw_feature_order, data):
            bucket_size = int(np.floor(data.shape[0] / self.bucket_num))
            split_position = list()
            self.split_value = list()

            for feature_idx in range(len(raw_feature_order)):
                bucketized_feature_order = bucketize(
                    raw_feature_order[feature_idx], bucket_size,
                    self.bucket_num)

                # Save split positions (instance number within buckets)
                # We exclude the endpoints to avoid empty sub-trees
                _split_position = list()
                _split_value = dict()
                accumu_num = 0
                for bucket_idx, each_bucket in enumerate(
                        bucketized_feature_order):
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

            extra_info = {'split_position': split_position}

            return {
                'feature_order': raw_feature_order,
                'extra_info': extra_info
            }

        def _get_feature_order_info(self, data):
            num_of_feature = data.shape[1]
            feature_order = [0] * num_of_feature
            for i in range(num_of_feature):
                feature_order[i] = data[:, i].argsort()
            return self._bucketize(feature_order, data)

        def get_feature_value(self, feature_idx, value_idx):
            if not hasattr(self, 'split_value') or self.split_value is None:
                return super().get_feature_value(feature_idx=feature_idx,
                                                 value_idx=value_idx)

            return self.split_value[feature_idx][value_idx]

        def get_abs_value_idx(self, feature_idx, value_idx):
            if self.extra_info is not None and self.extra_info.get(
                    'split_position', None) is not None:
                return self.extra_info['split_position'][feature_idx][value_idx
                                                                      - 1]
            else:
                return value_idx

        def _compute_for_node(self, tree_num, node_num):

            # All the nodes have been traversed
            if node_num >= 2**self.model.max_depth - 1:
                self._predict(tree_num)
                return 'train_finish', None
            elif self.model[tree_num][node_num].status == 'off':
                return self._compute_for_node(tree_num, node_num + 1)
            # The leaf node
            elif node_num >= 2**(self.model.max_depth - 1) - 1:
                self._set_weight_and_status(tree_num, node_num)
                return self._compute_for_node(tree_num, node_num + 1)
            # Calculate sum of grad and hess based on the encrypted results
            else:
                en_grad = [
                    self.public_key.encrypt(x)
                    for x in self.model[tree_num][node_num].grad
                ]
                if self.model[tree_num][node_num].hess is not None:
                    en_hess = [
                        self.public_key.encrypt(x)
                        for x in self.model[tree_num][node_num].hess
                    ]
                    en_indicator = None
                else:
                    en_indicator = [
                        self.public_key.encrypt(x)
                        for x in self.model[tree_num][node_num].indicator
                    ]
                    en_hess = None
                results = (en_grad, en_hess, en_indicator, tree_num, node_num)

                return 'call_for_local_gain', results

        def _get_best_gain(self, tree_num, node_num, grad, hess, indicator):
            # We can only get partial sum since the grad/hess is encrypted

            if self.merged_feature_order is None:
                self.merged_feature_order = self.client_feature_order
            if self.extra_info is None:
                self.extra_info = self.client_extra_info

            feature_num = len(self.merged_feature_order)
            split_position = self.extra_info.get('split_position')
            sum_of_grad = list()
            sum_of_hess = list()
            sum_of_indicator = list()

            for feature_idx in range(feature_num):
                ordered_g, ordered_h, ordered_indicator = self._get_ordered_gh(
                    tree_num, node_num, feature_idx, grad, hess, indicator)
                start_idx = 0
                _sum_of_grad = list()
                _sum_of_hess = list()
                _sum_of_indicator = list()
                for value_idx in split_position[feature_idx]:
                    _sum_of_grad.append(np.sum(ordered_g[start_idx:value_idx]))
                    if ordered_h is not None:
                        _sum_of_hess.append(
                            np.sum(ordered_h[start_idx:value_idx]))
                    else:
                        _sum_of_indicator.append(
                            np.sum(ordered_indicator[start_idx:value_idx]))
                    start_idx = value_idx
                _sum_of_grad.append(np.sum(ordered_g[start_idx:]))
                sum_of_grad.append(_sum_of_grad)
                if ordered_h is not None:
                    _sum_of_hess.append(np.sum(ordered_h[start_idx:]))
                    sum_of_hess.append(_sum_of_hess)
                    sum_of_indicator = None
                else:
                    _sum_of_indicator.append(
                        np.sum(ordered_indicator[start_idx:]))
                    sum_of_indicator.append(_sum_of_indicator)
                    sum_of_hess = None

            results = {
                'sum_of_grad': sum_of_grad,
                'sum_of_hess': sum_of_hess,
                'sum_of_indicator': sum_of_indicator
            }
            return False, results, None

        def get_best_gain_from_msg(self, msg, tree_num=None, node_num=None):
            client_has_max_gain = None
            best_gain = None
            split_ref = {}
            for client_id, local_gain in msg.items():
                _, _, split_info = local_gain
                sum_of_grad = split_info['sum_of_grad']
                sum_of_hess = split_info['sum_of_hess']
                sum_of_indicator = split_info['sum_of_indicator']
                for feature_idx in range(len(sum_of_grad)):
                    grad = [
                        self.private_key.decrypt(x)
                        for x in sum_of_grad[feature_idx]
                    ]
                    if sum_of_hess is not None:
                        hess = [
                            self.private_key.decrypt(x)
                            for x in sum_of_hess[feature_idx]
                        ]
                        indicator = None
                    else:
                        indicator = [
                            self.private_key.decrypt(x)
                            for x in sum_of_indicator[feature_idx]
                        ]
                        hess = None

                    for value_idx in range(1, len(grad)):
                        gain = self.model[tree_num].cal_gain(
                            grad, hess, value_idx, indicator)

                        if best_gain is None or gain > best_gain:
                            client_has_max_gain = client_id
                            best_gain = gain
                            split_ref['feature_idx'] = feature_idx
                            split_ref['value_idx'] = value_idx

            return best_gain, client_has_max_gain, split_ref

    return LabelProtectedTrainer(model, data, device, config, monitor)
