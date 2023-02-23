import numpy as np

from federatedscope.vertical_fl.trainer import VerticalTrainer
from federatedscope.vertical_fl.dataloader.utils import VerticalDataSampler
from federatedscope.vertical_fl.loss.utils import get_vertical_loss


class RandomForestTrainer(VerticalTrainer):
    def __init__(self, model, data, device, config, monitor):
        super(RandomForestTrainer, self).__init__(model, data, device, config,
                                                  monitor)

    def _init_for_train(self):
        self.eta = 1.0
        self.model.set_task_type(self.cfg.criterion.type)
        self.dataloader = VerticalDataSampler(
            data=self.data['train'],
            replace=True,
            use_full_trainset=False,
            feature_frac=self.cfg.vertical.feature_subsample_ratio)
        self.criterion = get_vertical_loss(loss_type=self.cfg.criterion.type,
                                           model_type=self.cfg.model.type)

    def _compute_for_root(self, tree_num):
        node_num = 0
        self.model[tree_num][node_num].label = self.batch_y
        self.model[tree_num][node_num].indicator = np.ones(len(self.batch_y))
        return self._compute_for_node(tree_num, node_num=node_num)

    def _get_ordered_indicator_and_label(self, tree_num, node_num,
                                         feature_idx):
        order = self.merged_feature_order[feature_idx]
        ordered_indicator = self.model[tree_num][node_num].indicator[order]
        ordered_label = self.model[tree_num][node_num].label[order]
        return ordered_indicator, ordered_label

    def _get_best_gain(self, tree_num, node_num, grad=None, hess=None):
        if self.cfg.criterion.type == 'CrossEntropyLoss':
            default_gain = 1
        elif 'Regression' in self.cfg.criterion.type:
            default_gain = float('inf')
        else:
            raise ValueError

        split_ref = {'feature_idx': None, 'value_idx': None}
        best_gain = default_gain
        feature_num = len(self.merged_feature_order)
        split_position = None
        if self.extra_info is not None:
            split_position = self.extra_info.get('split_position', None)

        activate_idx = [
            np.nonzero(self.model[tree_num][node_num].indicator[order])[0]
            for order in self.merged_feature_order
        ]
        activate_idx = np.asarray(activate_idx)
        if split_position is None:
            # The left/right sub-tree cannot be empty
            split_position = activate_idx[:, 1:]

        for feature_idx in range(feature_num):
            if len(split_position[feature_idx]) == 0:
                continue
            ordered_indicator, ordered_label = \
                self._get_ordered_indicator_and_label(
                    tree_num, node_num, feature_idx)
            order = self.merged_feature_order[feature_idx]
            for value_idx in split_position[feature_idx]:
                if self.model[tree_num].check_empty_child(
                        node_num, value_idx, order):
                    continue
                gain = self.model[tree_num].cal_gain(value_idx, ordered_label,
                                                     ordered_indicator)
                if gain < best_gain:
                    best_gain = gain
                    split_ref['feature_idx'] = feature_idx
                    split_ref['value_idx'] = value_idx

        return best_gain < default_gain, split_ref, best_gain
