import copy
from typing import Dict, List


def wrap_regularized_optimizer(base_optimizer, regular_weight):
    base_optimizer_type = type(base_optimizer)
    internal_base_optimizer = copy.copy(
        base_optimizer)  # shallow copy to link the underlying model para

    class ParaRegularOptimizer(base_optimizer_type):
        """
        Regularization-based optimizer wrapper
        """
        def __init__(self, base_optimizer, regular_weight):
            # inherit all the attributes of base optimizer
            self.__dict__.update(base_optimizer.__dict__)

            # attributes used in the wrapper
            self.optimizer = base_optimizer  # internal torch optimizer
            self.param_groups = self.optimizer.param_groups  # link the para
            # of internal optimizer with the wrapper
            self.regular_weight = regular_weight
            self.compared_para_groups = None

        def set_compared_para_group(self, compared_para_dict: List[Dict]):
            if not (isinstance(compared_para_dict, list)
                    and isinstance(compared_para_dict[0], dict)
                    and 'params' in compared_para_dict[0]):
                raise ValueError(
                    "compared_para_dict should be a torch style para group, "
                    "i.e., list[dict], "
                    "in which the dict stores the para with key `params`")
            self.compared_para_groups = copy.deepcopy(compared_para_dict)

        def reset_compared_para_group(self, target=None):
            # by default, del stale compared_para to free memory
            self.compared_para_groups = target

        def regularize_by_para_diff(self):
            """
                before optim.step(), regularize the gradients based on para
                diff
            """
            for group, compared_group in zip(self.param_groups,
                                             self.compared_para_groups):
                for p, compared_weight in zip(group['params'],
                                              compared_group['params']):
                    if p.grad is not None:
                        if compared_weight.device != p.device:
                            # For Tensor, the to() is not in-place operation
                            compared_weight = compared_weight.to(p.device)
                        p.grad.data = p.grad.data + self.regular_weight * (
                            p.data - compared_weight.data)

        def step(self):
            self.regularize_by_para_diff()  # key action
            self.optimizer.step()

    return ParaRegularOptimizer(internal_base_optimizer, regular_weight)
