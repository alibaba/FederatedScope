import logging
import collections

from federatedscope.core.auxiliaries.criterion_builder import get_criterion
from federatedscope.core.auxiliaries.model_builder import \
    get_trainable_para_names
from federatedscope.core.auxiliaries.regularizer_builder import get_regularizer
from federatedscope.core.auxiliaries.enums import MODE
from federatedscope.core.auxiliaries.utils import calculate_batch_epoch_num
from federatedscope.core.interface.base_data import ClientData

logger = logging.getLogger(__name__)


class LifecycleDict(dict):
    """A customized dict that provides lifecycle management
    Arguments:
        init_dict: initialized dict
    """
    __delattr__ = dict.__delitem__

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError("Attribute {} is not found".format(item))

    def __init__(self, init_dict=None):
        if init_dict is not None:
            super(LifecycleDict, self).__init__(init_dict)
        self.lifecycles = collections.defaultdict(set)

    def __setattr__(self, key, value):
        if isinstance(value, CtxVar):
            self.lifecycles[value.lifecycle].add(key)
            super(LifecycleDict, self).__setitem__(key, value.obj)
        else:
            super(LifecycleDict, self).__setitem__(key, value)

    def clear(self, lifecycle):
        keys = list(self.lifecycles[lifecycle])
        for key in keys:
            if key in self:
                del self[key]
            self.lifecycles[lifecycle].remove(key)


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
        - The variables within an instance of class `Context`
        can be set/get as an attribute.
        ```
        ctx.${NAME_VARIABLE} = ${VALUE_VARIABLE}
        ```
        where `${NAME_VARIABLE}` and `${VALUE_VARIABLE}`
        is the name and value of the variable.

        - To achieve automatically lifecycle management, you can
        wrap the variable with `CtxVar` and a lifecycle parameter
        as follows
        ```
        ctx.${NAME_VARIABLE} = CtxVar(${VALUE_VARIABLE}, ${LFECYCLE})
        ```
        The parameter `${LFECYCLE}` can be chosen from `LIFECYCLE.BATCH`,
        `LIFECYCLE.EPOCH` and `LIFECYCLE.ROUTINE`.
        Then the variable `ctx.${NAME_VARIABLE}` will be deleted at
        the end of the corresponding stage
            - `LIFECYCLE.BATCH`: the variables will
            be deleted after running a batch
            - `LIFECYCLE.EPOCH`: the variables will be
            deleted after running a epoch
            - `LIFECYCLE.ROUTINE`: the variables will be
            deleted after running a routine
        More details please refer to our
        [tutorial](https://federatedscope.io/docs/trainer/).

        - Context also maintains some special variables across
        different routines, like
            - cfg
            - model
            - data
            - device
            - ${split}_data: the dataset object of data split
            named `${split}`
            - ${split}_loader: the data loader object of data
            split named `${split}`
            - num_${split}_data: the number of examples within
            the dataset named `${split}`
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

        self.lifecycles = collections.defaultdict(set)

        if init_attr:
            # setup static variables for training/evaluation
            self.setup_vars()

    def setup_vars(self):
        if self.cfg.backend == 'torch':
            self.trainable_para_names = get_trainable_para_names(self.model)
            self.criterion = get_criterion(self.cfg.criterion.type,
                                           self.device)
            self.regularizer = get_regularizer(self.cfg.regularizer.type)
            self.grad_clip = self.cfg.grad.grad_clip
            if isinstance(self.data, ClientData):
                self.data.setup(self.cfg)
        elif self.cfg.backend == 'tensorflow':
            self.trainable_para_names = self.model.trainable_variables()
            self.criterion = None
            self.regularizer = None
            self.optimizer = None
            self.grad_clip = None

        # Process training data
        if self.get('train_data', None) is not None or self.get(
                'train_loader', None) is not None:
            # Calculate the number of update steps during training given the
            # local_update_steps
            self.num_train_batch, self.num_train_batch_last_epoch, \
                self.num_train_epoch, self.num_total_train_batch = \
                calculate_batch_epoch_num(
                    self.cfg.train.local_update_steps,
                    self.cfg.train.batch_or_epoch, self.num_train_data,
                    self.cfg.data.batch_size, self.cfg.data.drop_last)

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

    def track_mode(self, mode):
        self.mode_stack.append(mode)
        self.cur_mode = self.mode_stack[-1]
        self.change_mode(self.cur_mode)

    def reset_mode(self):
        self.mode_stack.pop()
        self.cur_mode = self.mode_stack[-1] if len(
            self.mode_stack) != 0 else None
        if len(self.mode_stack) != 0:
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
                raise ValueError(f"No {target_split_name}_data or"
                                 f" {target_split_name}_loader in the trainer")
        else:
            return True


class CtxVar(object):
    """Basic variable class
    Arguments:
        lifecycle: specific lifecycle of the attribute
    """

    LIEFTCYCLES = ["batch", "epoch", "routine", None]

    def __init__(self, obj, lifecycle=None):
        assert lifecycle in CtxVar.LIEFTCYCLES
        self.obj = obj
        self.lifecycle = lifecycle


def lifecycle(lifecycle):
    """Manage the lifecycle of the variables within context,
    and blind these operations from user.
    Args:
        lifecycle: the type of lifecycle, choose from "batch/epoch/routine"
    """
    if lifecycle == "routine":

        def decorate(func):
            def wrapper(self, mode, hooks_set, dataset_name=None):
                self.ctx.track_mode(mode)
                self.ctx.track_split(dataset_name or mode)

                res = func(self, mode, hooks_set, dataset_name)

                # Clear the variables at the end of lifecycles
                self.ctx.clear(lifecycle)

                # rollback the model and data_split
                self.ctx.reset_mode()
                self.ctx.reset_split()

                # Move the model into CPU to avoid memory leak
                self.discharge_model()

                return res

            return wrapper
    else:

        def decorate(func):
            def wrapper(self, *args, **kwargs):
                res = func(self, *args, **kwargs)
                # Clear the variables at the end of lifecycles
                self.ctx.clear(lifecycle)
                return res

            return wrapper

    return decorate
