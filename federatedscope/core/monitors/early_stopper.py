import operator
import numpy as np


# TODO: make this as a sub-module of monitor class
class EarlyStopper(object):
    """
    Track the history of metric (e.g., validation loss), \
    check whether should stop (training) process if the metric doesn't \
    improve after a given patience.

    Args:
        patience (int): (Default: 5) How long to wait after last time the \
            monitored metric improved. Note that the \
            ``actual_checking_round = patience * cfg.eval.freq``
        delta (float): (Default: 0) Minimum change in the monitored metric to \
            indicate an improvement.
        improve_indicator_mode (str): Early stop when no improve to \
            last ``patience`` round, in ``['mean', 'best']``
    """
    def __init__(self,
                 patience=5,
                 delta=0,
                 improve_indicator_mode='best',
                 the_larger_the_better=True):
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
        self.early_stopped = False
        self.the_larger_the_better = the_larger_the_better
        self.delta = delta
        self.improve_indicator_mode = improve_indicator_mode
        # For expansion usages of comparisons
        self.comparator = operator.lt
        self.improvement_operator = operator.add

    def __track_and_check_dummy(self, new_result):
        """
        Dummy stopper, always return false

        Args:
            new_result:

        Returns:
            False
        """
        self.early_stopped = False
        return self.early_stopped

    def __track_and_check_best(self, history_result):
        """
        Tracks the best result and checks whether the patience is exceeded.

        Args:
            history_result: results of all evaluation round

        Returns:
            Bool: whether stop
        """
        new_result = history_result[-1]
        if self.best_metric is None:
            self.best_metric = new_result
        elif not self.the_larger_the_better and self.comparator(
                self.improvement_operator(self.best_metric, -self.delta),
                new_result):
            # add(best_metric, -delta) < new_result
            self.counter_no_improve += 1
        elif self.the_larger_the_better and self.comparator(
                new_result,
                self.improvement_operator(self.best_metric, self.delta)):
            # new_result < add(best_metric, delta)
            self.counter_no_improve += 1
        else:
            self.best_metric = new_result
            self.counter_no_improve = 0

        self.early_stopped = self.counter_no_improve >= self.patience
        return self.early_stopped

    def __track_and_check_mean(self, history_result):
        new_result = history_result[-1]
        if len(history_result) > self.patience:
            if not self.the_larger_the_better and self.comparator(
                    self.improvement_operator(
                        np.mean(history_result[-self.patience - 1:-1]),
                        -self.delta), new_result):
                self.early_stopped = True
            elif self.the_larger_the_better and self.comparator(
                    new_result,
                    self.improvement_operator(
                        np.mean(history_result[-self.patience - 1:-1]),
                        self.delta)):
                self.early_stopped = True
        else:
            self.early_stopped = False

        return self.early_stopped

    def track_and_check(self, new_result):
        """
        Checks the new result and if it improves it returns True.

        Args:
            new_result: new evaluation result

        Returns:
            Bool: whether stop
        """

        track_method = self.__track_and_check_dummy  # do nothing
        if self.patience == 0:
            track_method = self.__track_and_check_dummy
        elif self.improve_indicator_mode == 'best':
            track_method = self.__track_and_check_best
        elif self.improve_indicator_mode == 'mean':
            track_method = self.__track_and_check_mean

        return track_method(new_result)
