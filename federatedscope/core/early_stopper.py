import operator
import numpy as np


class EarlyStopper(object):
    """
        Track the history of metric (e.g., validation loss),
        check whether should stop (training) process if the metric doesn't improve after a given patience.
    """
    def __init__(self,
                 patience=5,
                 delta=0,
                 improve_indicator_mode='best',
                 the_smaller_the_better=True):
        """
        Args:
            patience (int): How long to wait after last time the monitored metric improved.
                            Note that the actual_checking_round = patience * cfg.eval.freq
                            Default: 5
            delta (float): Minimum change in the monitored metric to indicate an improvement.
                            Default: 0
            improve_indicator_mode (str): Early stop when no improve to last `patience` round, in ['mean', 'best']
        """
        assert 0 <= patience == int(
            patience
        ), "Please use a non-negtive integer to indicate the patience"
        assert delta >= 0, "Please use a positive value to indicate the change"
        assert improve_indicator_mode in [
            'mean', 'best'
        ], "Please make sure `improve_indicator_mode` is 'mean' or 'best']"

        self.patience = patience
        self.counter_no_improve = 0
        self.best_metric = None
        self.early_stop = False
        self.the_smaller_the_better = the_smaller_the_better
        self.delta = delta
        self.improve_indicator_mode = improve_indicator_mode
        # For expansion usages of comparisons
        self.comparator = operator.lt
        self.improvement_operator = operator.add

    def track_and_check_dummy(self, new_result):
        return False

    def track_and_check_best(self, history_result):
        new_result = history_result[-1]
        if self.best_metric is None:
            self.best_metric = new_result
        elif self.the_smaller_the_better and self.comparator(
                self.improvement_operator(self.best_metric, -self.delta),
                new_result):
            # by default: add(val_loss, -delta) < new_result
            self.counter_no_improve += 1
        elif not self.the_smaller_the_better and self.comparator(
                self.improvement_operator(self.best_metric, self.delta),
                new_result):
            # typical case: add(eval_score, delta) > new_result
            self.counter_no_improve += 1
        else:
            self.best_metric = new_result
            self.counter_no_improve = 0

        return self.counter_no_improve >= self.patience

    def track_and_check_mean(self, history_result):
        new_result = history_result[-1]
        if len(history_result) > self.patience:
            if self.the_smaller_the_better and self.comparator(
                    self.improvement_operator(
                        np.mean(history_result[-self.patience - 1:-1]),
                        -self.delta), new_result):
                return True
            elif not self.the_smaller_the_better and self.comparator(
                    self.improvement_operator(
                        np.mean(history_result[-self.patience - 1:-1]),
                        self.delta), new_result):
                return True
        return False

    def track_and_check(self, new_result):
        if self.patience:
            if self.improve_indicator_mode == 'best':
                return self.track_and_check_best(new_result)
            elif self.improve_indicator_mode == 'mean':
                return self.track_and_check_mean(new_result)
        return self.track_and_check_dummy(new_result)