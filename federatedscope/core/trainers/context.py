import math
import logging
import collections

from federatedscope.core.auxiliaries.criterion_builder import get_criterion
from federatedscope.core.auxiliaries.model_builder import \
    get_trainable_para_names
from federatedscope.core.auxiliaries.regularizer_builder import get_regularizer
from federatedscope.core.auxiliaries.eunms import MODE
from federatedscope.core.auxiliaries.utils import calculate_batch_epoch_num

logger = logging.getLogger(__name__)

class LifecycleDict(dict):
    """A customized dict that provides lifecycle management
    Arguments:
        init_dict: initialized dict
    """
    __delattr__ = dict.__delitem__
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

    def __init__(self, init_dict=None):
        if init_dict is not None:
            super(LifecycleDict, self).__init__(init_dict)
        self.lifecycles = collections.defaultdict(set)

    def __setitem__(self, key, value):
        if isinstance(value, CtxVar):
            self.lifecycles[value.lifecycle].add(key)
        super(LifecycleDict, self).__setitem__(key, value)

    def clear(self, lifecycle):
        for var in self.lifecycles[lifecycle]:
            if hasattr(self[var], "clear"):
                self[var].clear()
            else:
                del self[var]


class Context(LifecycleDict):
    """Record and pass variables among different hook functions
    Arguments:
        model: training model
        cfg: config
        data (dict): a dict contains train/val/test dataset or dataloader
        device: running device
        init_dict (dict): a dict used to initialize the instance of Context
        init_attr (bool): if set up the static variables
    Note:
        - There are two ways to set/get the attributes within an instance `ctx`:
            - `ctx.${NAME}`: the variable is assigned to `ctx` as an attribute without additional operations
            - `ctx.var.${NAME}`: the variable is stored in `ctx["var"][${ctx.cur_mode}_${ctx.cur_split}][${NAME}]`, \
            and is only accessible when `ctx.cur_mode` and `ctx.cur_data_split` is consistent.
        The setting of `ctx.var.${NAME}` allows you to nest test routine within the training routine, and the record \
        variables with the same name will be stored according to current mode (`ctx.cur_mode`) and current data split \
        (`ctx.cur_split`), which won't cover each other.

        - While Context also maintain some special variables across different routines, like
            - cfg
            - model
            - data
            - device
            - ${split}_data: the dataset object of data split named `${split}`
            - ${split}_loader: the data loader object of data split named `${split}`
            - num_${split}_data: the number of examples within the dataset named `${split}`
    """

    def __init__(self,
                 model,
                 cfg,
                 data=None,
                 device=None,
                 init_dict=None,
                 init_attr=True):
        super(Context, self).__init__(init_dict)

        self.cfg = cfg
        self.model = model
        self.data = data
        self.device = device

        self.cur_mode = None
        self.mode_stack = list()

        self.cur_split = None
        self.split_stack = list()

        self.var = collections.defaultdict(LifecycleDict)
        self.lifecycles = collections.defaultdict(set)

        if init_attr:
            # setup static variables for training/evaluation
            self.setup_vars()

    def __getattr__(self, item):
        try:
            if item == 'var':
                return self["var"]["{}_{}".format(self.cur_mode, self.cur_split)]
            else:
                return self[item]
        except KeyError:
            raise AttributeError(item)

    def setup_vars(self):
        if self.cfg.backend == 'torch':
            self.trainable_para_names = get_trainable_para_names(self.model)
            self.criterion = get_criterion(self.cfg.criterion.type,
                                           self.device)
            self.regularizer = get_regularizer(self.cfg.regularizer.type)
            self.grad_clip = self.cfg.grad.grad_clip
        elif self.cfg.backend == 'tensorflow':
            self.trainable_para_names = self.model.trainable_variables()
            self.criterion = None
            self.regularizer = None
            self.optimizer = None
            self.grad_clip = None

    def get_variable(self, mode, data_split, key):
        """To support the access of variables that doesn't belong the current mode
        Args:
            mode: which mode that the variable belongs to
            data_split: which data split that the variable belongs to
            key: the name of variable
        Returns: the wanted variable
        """
        return self["var"][f"{mode}_{data_split}"][key]

    def init_routine(self):
        if self.cur_mode in [MODE.TEST, MODE.VAL]:
            steps, batch_or_epoch = 1, 'epoch'
        else:
            cfg_mode = self.cfg.get(self.cur_mode)
            steps, batch_or_epoch = cfg_mode.local_update_steps, cfg_mode.batch_or_epoch

        num_data = self.get(f'num_{self.cur_split}_data')

        self.var.num_batch, self.var.num_batch_last_epoch, self.var.num_epoch, self.var.num_total_batch = [CtxVar(_, 'routine') for _ in calculate_batch_epoch_num(
            steps=steps,
            batch_or_epoch=batch_or_epoch,
            num_data=num_data,
            batch_size=self.cfg.data.batch_size,
            drop_last=self.cfg.data.drop_last
        )]

    def track_mode(self, mode):
        self.mode.append(mode)
        self.cur_mode = self.mode[-1]
        self.change_mode(self.cur_mode)

    def reset_mode(self):
        self.mode.pop()
        self.cur_mode = self.mode[-1] if len(self.mode) != 0 else None
        if len(self.mode) != 0:
            self.change_mode(self.cur_mode)

    def change_mode(self, mode):
        # change state
        if self.cfg.backend == 'torch':
            getattr(
                self.model, 'train'
                if mode == MODE.TRAIN or mode == MODE.FINETUNE else 'eval')()
        else:
            pass

    def track_split(self, dataset):
        # stack-style to enable mixture usage such as evaluation on train
        # dataset
        self.split_stack.append(dataset)
        self.cur_split = self.split_stack[-1]

    def reset_split(self):
        self.split_stack.pop()
        self.cur_split = self.split_stack[-1] if \
            len(self.split_stack) != 0 else None

    def check_split(self, target_split_name, skip=False):
        if self.get(f"{target_split_name}_data") is None and self.get(
                f"{target_split_name}_loader") is None:
            if skip:
                logger.warning(
                    f"No {target_split_name}_data or"
                    f" {target_split_name}_loader in the trainer, "
                    f"will skip evaluation"
                    f"If this is not the case you want, please check "
                    f"whether there is typo for the name")
                return False
            else:
                raise ValueError(
                    f"No {target_split_name}_data or"
                    f" {target_split_name}_loader in the trainer")
        else:
            return True


class CtxVar(object):
    """Basic variable class
    Arguments:
        lifecycle: specific lifecycle of the attribute
    """

    LIEFTCYCLES = ["batch", "epoch", "routine", None]

    def __init__(self, lifecycle=None, end_func=None):
        assert lifecycle in CtxVar.LIEFTCYCLES
        self.lifecycle = lifecycle
        self.efunc = end_func

    def clear(self):
        if self.efunc is not None:
            self.efunc(self.obj)


def lifecycle(lifecycle):
    """Manage the lifecycle of the variables within context, and blind these operations from user.
    Args:
        lifecycle: the type of lifecycle, choose from "batch/epoch/routine"
    """
    if lifecycle == "routine":

        def decorate(func):
            def wrapper(self, mode, dataset_name=None):
                self.ctx.track_mode(mode)
                self.ctx.track_split(dataset_name or mode)

                res = func(self, mode, dataset_name)

                # Clear the variables at the end of lifecycles
                self.ctx.clear(lifecycle)

                self.ctx.reset_mode()
                self.ctx.reset_split()

                return res

            return wrapper
    else:

        def decorate(func):
            def wrapper(self):
                res = func(self)
                # Clear the variables at the end of lifecycles
                self.ctx.clear(lifecycle)
                return res

            return wrapper

    return decorate