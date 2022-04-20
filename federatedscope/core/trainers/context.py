import collections
import math

from federatedscope.core.auxiliaries.criterion_builder import get_criterion
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.model_builder import get_trainable_para_names
from federatedscope.core.auxiliaries.regularizer_builder import get_regularizer


class LifecycleDict(dict):
    """A customized dict that provides lifecycle management

    Arguments:
        init_dict: initialized dict
    """
    __delattr__ = dict.__delitem__

    def __init__(self, init_dict=None):
        if init_dict is not None:
            super(LifecycleDict, self).__init__(init_dict)
        self.lifecycles = collections.defaultdict(set)

    def __getitem__(self, item):
        value = super(LifecycleDict, self).__getitem__(item)
        if isinstance(value, CtxReferVar):
            return value.obj
        else:
            return value

    def __setitem__(self, key, value):
        if isinstance(value, BasicCtxVar):
            self.lifecycles[value.lifecycle].add(key)
        super(LifecycleDict, self).__setitem__(key, value)

    def get(self, *args, **kwargs):
        value = super(LifecycleDict, self).get(*args, **kwargs)
        if isinstance(value, CtxReferVar):
            return value.obj
        else:
            return value

    def __delattr__(self, item):
        self.__delitem__(item)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __getattr__(self, item):
        return self.__getitem__(item)

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
        There are two ways to set/get the variables within an instance `ctx`:
            - `ctx.${NAME}`: the variable is assigned to `ctx` as an attribute without additional operations
            - `ctx.mode.${NAME}`: the variable is stored in `ctx["mode"][ctx.cur_mode][${NAME}]`, and is only accessible \
            when `ctx.cur_mode` is correct.
        The setting of `ctx.mode.${NAME}` means you can nest test routine in the training routine. The record \
        variables with the same name will be stored according to `ctx.cur_mode` and won't influence each other. For \
        now, the `Context` class only permits nested routine with different `ctx.cur_mode`.
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
        self.mode = collections.defaultdict(LifecycleDict)

        self.lifecycles = collections.defaultdict(set)

        self.cur_data_split = None

        if init_attr:
            # setup static variables for training/evaluation
            self._setup_vars()

    def __getattr__(self, item):
        try:
            if item == "mode":
                return self["mode"][self.cur_mode]
            else:
                return self[item]
        except KeyError:
            raise AttributeError(item)

    def _setup_vars(self):
        if self.cfg.backend == 'torch':
            self.trainable_para_names = get_trainable_para_names(self.model)
            self.criterion = get_criterion(self.cfg.criterion.type,
                                           self.device)
            self.regularizer = get_regularizer(self.cfg.regularizer.type)
            self.optimizer = get_optimizer(
                self.cfg.optimizer.type,
                self.model,
                self.cfg.optimizer.lr,
                weight_decay=self.cfg.optimizer.weight_decay)
            self.grad_clip = self.cfg.optimizer.grad_clip
        elif self.cfg.backend == 'tensorflow':
            self.trainable_para_names = self.model.trainable_variables()
            self.criterion = None
            self.regularizer = None
            self.optimizer = None
            self.grad_clip = None

        self.cur_data_splits_used_by_routine = list()

        # Process training data
        if self.train_data is not None or self.train_loader is not None:
            # Calculate the number of update steps during training given the local_update_steps
            num_train_batch, num_train_batch_last_epoch, num_train_epoch, num_total_train_batch = self.pre_calculate_batch_epoch_num(
                self.cfg.federate.local_update_steps)

            self.num_train_epoch = num_train_epoch
            self.num_train_batch = num_train_batch
            self.num_train_batch_last_epoch = num_train_batch_last_epoch
            self.num_total_train_batch = num_total_train_batch

        # Process evaluation data
        for mode in ["val", "test"]:
            setattr(self, "num_{}_epoch".format(mode), 1)
            if self.get("{}_data".format(mode)) is not None or self.get(
                    "{}_loader".format(mode)) is not None:
                setattr(
                    self, "num_{}_batch".format(mode),
                    getattr(self, "num_{}_data".format(mode)) //
                    self.cfg.data.batch_size +
                    int(not self.cfg.data.drop_last and bool(
                        getattr(self, "num_{}_data".format(mode)) %
                        self.cfg.data.batch_size)))

    def get_variable(self, mode, key):
        """To support the access of variables that doesn't belong the current mode

        Args:
            mode: which mode that the variable belongs to
            key: the name of variable

        Returns: the wanted variable

        """
        return self[mode][key]

    def pre_calculate_batch_epoch_num(self, local_update_steps):
        num_train_batch = self.num_train_data // self.cfg.data.batch_size + int(
            not self.cfg.data.drop_last
            and bool(self.num_train_data % self.cfg.data.batch_size))
        if self.cfg.federate.batch_or_epoch == "epoch":
            num_train_epoch = local_update_steps
            num_train_batch_last_epoch = num_train_batch
            num_total_train_batch = local_update_steps * num_train_batch
        else:
            num_train_epoch = math.ceil(local_update_steps / num_train_batch)
            num_train_batch_last_epoch = local_update_steps % num_train_batch
            num_total_train_batch = local_update_steps
        return num_train_batch, num_train_batch_last_epoch, num_train_epoch, num_total_train_batch

    def append_mode(self, mode):
        if mode in self.mode_stack:
            raise RuntimeError(
                "FederatedScope doesn't support nested routine with the same mode {}, variables could be covered.".format(
                    mode))
        self.mode_stack.append(mode)
        self.cur_mode = self.mode_stack[-1]
        self.change_mode(self.cur_mode)
        if self.cur_mode not in self:
            self[self.cur_mode] = dict()

    def pop_mode(self):
        self.mode_stack.pop()
        self.cur_mode = self.mode_stack[-1] if len(self.mode_stack) != 0 else None
        if len(self.mode_stack) != 0:
            self.change_mode(self.cur_mode)

    def change_mode(self, mode):
        # change state
        if self.cfg.backend == 'torch':
            getattr(self.model, mode if mode == 'train' else 'eval')()
        else:
            pass

    def track_used_dataset(self, dataset):
        # stack-style to enable mixture usage such as evaluation on train dataset
        self.cur_data_splits_used_by_routine.append(dataset)
        self.cur_data_split = self.cur_data_splits_used_by_routine[-1]

    def reset_used_dataset(self):
        self.cur_data_splits_used_by_routine.pop()
        self.cur_data_split = self.cur_data_splits_used_by_routine[-1] if \
            len(self.cur_data_splits_used_by_routine) != 0 else None

    def clear(self, lifecycle):
        """Clear the variables at the end of their lifecycle

        Args:
            lifecycle: the type of lifecycle

        Returns:

        """
        attrs = list(self.lifecycles[lifecycle])
        # Clear general attributes
        for var in attrs:
            if hasattr(self[var], "clear"):
                self[var].clear()
            else:
                del self[var]
            self.lifecycles[lifecycle].remove(var)
        # Also clear the attributes under the current mode
        self.mode.clear(lifecycle)



class BasicCtxVar(object):
    """Basic variable class

    Arguments:
        lifecycle: specific lifecycle of the attribute
    """

    LIEFTCYCLES = [
        "batch",
        "epoch",
        "routine",
        None
    ]

    def __init__(self, lifecycle=None):
        assert lifecycle in BasicCtxVar.LIEFTCYCLES

        self.lifecycle = lifecycle


class CtxReferVar(BasicCtxVar):
    """To store the reference variables with specific lifecycle and clear function, e.g. model, data, dataloader

    Arguments:
        obj: the stored obj
        lifecycle: the specific lifecycle of the variable
        efunc: a function that will be called when the lifecycle ends

    """
    def __init__(self, obj, lifecycle=None, efunc=None):
        super(CtxReferVar, self).__init__(lifecycle=lifecycle)
        self.obj = obj
        self.efunc = efunc

    def clear(self):
        if self.efunc is not None:
            self.efunc(self.obj)


def CtxStatsVar(init=0., lifecycle="routine"):
    """To store the statistic digits with specific lifecycle, e.g. loss_batch, loss_total. The type is the same with `init`

    Arguments:
        init: the initialized value
        lifecycle: the specific lifecycle of the variable

    """
    fcls = type(init)
    class TemplateVar(BasicCtxVar, fcls):
        def __new__(cls, init=0., *args, **kwargs):
            return super(TemplateVar, cls).__new__(cls, init)

        def __init__(self, init=0., lifecycle='routine'):
            BasicCtxVar.__init__(self, lifecycle=lifecycle)
            fcls.__init__(self)
    return TemplateVar(init, lifecycle)


def lifecycle(lifecycle):
    """Manage the lifecycle of the variables within context, and blind these operations from user.

    Args:
        lifecycle: the type of lifecycle, choose from "batch/epoch/routine"
    """
    if lifecycle == "routine":
        def decorate(func):
            def wrapper(self, mode, dataset_name=None):
                self.ctx.append_mode(mode)
                self.ctx.track_used_dataset(dataset_name or mode)

                res = func(self, mode, dataset_name)

                # Clear the variables at the end of lifecycles
                self.ctx.clear(lifecycle)

                self.ctx.pop_mode()
                self.ctx.reset_used_dataset()

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

