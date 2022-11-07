
import numpy as np

# For a client
class AdvancedComposition(object):
    pass


class PrivacyAccountantComposition(object):
    pass


class RenyiComposition(object):
    def __init__(self, sample_rate):
        self.orders = [1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64]

        self.budgets = np.zeros_like(self.orders)

        self.epsilon = 0

        # sampling rate
        # alpha rate

    def compose(self, scale):
        for i, order in enumerate(self.orders):
            epsilon = order / scale ** 2
            self.budgets[i] += epsilon
