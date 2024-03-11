import logging
import collections

from federatedscope.core.auxiliaries.criterion_builder import get_criterion
from federatedscope.core.auxiliaries.model_builder import \
    get_trainable_para_names
from federatedscope.core.auxiliaries.regularizer_builder import get_regularizer
from federatedscope.core.trainers.enums import MODE
from federatedscope.core.trainers.utils import calculate_batch_epoch_num

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
    """
    Record and pass variables among different hook functions.

    Arguments:
        model: training model
        cfg: config
        data (dict): a dict contains train/val/test dataset or dataloader
        device: running device
        init_dict (dict): a dict used to initialize the instance of Context
        init_attr (bool): if set up the static variables
    Note:
        - The variables within an instance of class `Context` can be set/get \
        as an attribute.
        ```
        ctx.${NAME_VARIABLE} = ${VALUE_VARIABLE}
        ```
        where ``${NAME_VARIABLE}`` and ``${VALUE_VARIABLE}``
        is the name and value of the variable.

        - To achieve automatically lifecycle management, you can \
        wrap the variable with ``CtxVar`` and a lifecycle parameter \
        as follows
        ```
        ctx.${NAME_VARIABLE} = CtxVar(${VALUE_VARIABLE}, ${LIFECYCLE})
        ```
        The parameter ``${LIFECYCLE}`` can be chosen from \
        ``LIFECYCLE.BATCH``, ``LIFECYCLE.EPOCH`` and ``LIFECYCLE.ROUTINE``. \
        Then the variable ``ctx.${NAME_VARIABLE}`` will be deleted at \
        the end of the corresponding stage
            - ``LIFECYCLE.BATCH``: the variables will \
            be deleted after running a batch
            - ``LIFECYCLE.EPOCH``: the variables will be \
            deleted after running a epoch
            - ``LIFECYCLE.ROUTINE``: the variables will be \
            deleted after running a routine
        More details please refer to our
        [tutorial](https://federatedscope.io/docs/trainer/).

        We classify and show the default attributes below:

        Data-related attributes
          - ``ctx.data``: the raw data (not split) the trainer holds
          - ``ctx.num_samples``: the number of samples used in training
          - ``ctx.train_data``, ``ctx.val_data``, ``ctx.test_data``: the \
          split data the trainer holds
          - ``ctx.train_loader``, ``ctx.val_loader``, ``ctx.test_loader``: \
          the DataLoader of each split data
          - ``ctx.num_train_data``, ``ctx.num_val_data``, \
          ``ctx.num_test_data``: the number of samples of  the split data \
          Model-related attributes
          - ``ctx.model``: the model used
          - ``ctx.models``: the multi models if use
          - ``ctx.mirrored_models``: the mirrored models
          - ``ctx.trainable_para_names``: the trainable parameter names of \
          the model
        Optimizer-related attributes
          - ``ctx.optimizer``: see ``torch.optim``
          - ``ctx.scheduler``: decays the learning rate of each parameter group
          - ``ctx.criterion``: loss/criterion function
          - ``ctx.regularizer``: regular terms
          - ``ctx.grad_clip``: gradient clipping
        Mode-related attributes
          - ``ctx.cur_mode``: mode of trainer, which is one of ``['train', \
          'val', 'test']``
          - ``ctx.mode_stack``: stack of mode, only used for switching mode
          - ``ctx.cur_split``: split of data, which is one of ``['train', \
          'val', 'test']`` (Note: use ``train`` data in ``test`` mode is \
          allowed)
          - ``ctx.split_stack``: stack of split, only used for switching data \
          split
        Metric-related attributes
          - ``ctx.loss_batch_total``: Loss of current batch
          - ``ctx.loss_regular_total``: Loss of regular term
          - ``ctx.y_true``:  true label of batch data
          - ``ctx.y_prob``: output of the model with batch data as input
          - ``ctx.ys_true``: true label of data
          - ``ctx.ys_prob``: output of the model
          - ``ctx.eval_metrics``: evaluation metrics calculated by \
          ``ctx.monitor``
          - ``ctx.monitor``: used for monitor trainer's behavior and statistics
        Other (statistics) attributes (@property, query from ``cfg`` if not \
        set)
          - ``ctx.cfg``: configuration of FL course
          - ``ctx.device``: current device, such as ``cpu`` and ``gpu0``.
          - ``ctx.num_train_batch_last_epoch``, \
          ``ctx.num_total_train_batch``: the number of batch
          - ``ctx.num_train_epoch``, ``ctx.num_val_epoch``, \
          ``ctx.num_test_epoch``: the number of epoch in each data split
          - ``ctx.num_train_batch``, ``ctx.num_val_batch``, \
          ``ctx.num_test_batch``: the number of batch in each data split
    """
    def __init__(self, model, cfg, data=None, device=None):
        super(Context, self).__init__({})

        self.cfg = cfg
        self.model = model
        self.data = data
        self.device = device

        self.cur_mode = None
        self.mode_stack = list()

        self.cur_split = None
        self.split_stack = list()

        self.lifecycles = collections.defaultdict(set)

        # Setup optimize-related context variable
        if self.cfg.backend == 'torch':
            self.trainable_para_names = get_trainable_para_names(self.model)
            # TODO: make `criterion` and `regularizer` @property and cached
            #  to compare whether changes happen
            self.criterion = get_criterion(self.cfg.criterion.type,
                                           self.device)
            self.regularizer = get_regularizer(self.cfg.regularizer.type)
            self.grad_clip = self.cfg.grad.grad_clip
            if self.cfg.federate.process_num > 1:
                self.model.to(self.device)
        elif self.cfg.backend == 'tensorflow':
            self.trainable_para_names = self.model.trainable_variables()
            self.criterion = None
            self.regularizer = None
            self.optimizer = None
            self.grad_clip = None

    # Train related property, query from `cfg` if not set
    @property
    def num_train_batch(self):
        if self.get('num_train_batch'):
            return self.get('num_train_batch')
        return self._calculate_batch_epoch_num(mode='train')[0]

    @property
    def num_train_batch_last_epoch(self):
        if self.get('num_train_batch_last_epoch'):
            return self.get('num_train_batch_last_epoch')
        return self._calculate_batch_epoch_num(mode='train')[1]

    @property
    def num_train_epoch(self):
        if self.get('num_train_epoch'):
            return self.get('num_train_epoch')
        return self._calculate_batch_epoch_num(mode='train')[2]

    @property
    def num_total_train_batch(self):
        if self.get('num_total_train_batch'):
            return self.get('num_total_train_batch')
        return self._calculate_batch_epoch_num(mode='train')[3]

    # Val related property, query from `cfg` if not set
    @property
    def num_val_batch(self):
        if self.get('num_val_batch'):
            return self.get('num_val_batch')
        return self._calculate_batch_epoch_num(mode='val')[0]

    @property
    def num_val_epoch(self):
        if self.get('num_val_epoch'):
            return self.get('num_val_epoch')
        return self._calculate_batch_epoch_num(mode='val')[2]

    # Test related property, query from `cfg` if not set
    @property
    def num_test_batch(self):
        if self.get('num_test_batch'):
            return self.get('num_test_batch')
        return self._calculate_batch_epoch_num(mode='test')[0]

    @property
    def num_test_epoch(self):
        if self.get('num_test_epoch'):
            return self.get('num_test_epoch')
        return self._calculate_batch_epoch_num(mode='test')[2]

    def _calculate_batch_epoch_num(self, mode='train'):
        if self.cur_mode is not None and self.cur_mode != mode:
            logger.warning(
                f'cur_mode `{self.cur_mode}` mismatch mode `{mode}`, '
                f'will use `{mode}` to calculate `ctx.var`.')
        if self.cur_split is None:
            logger.warning(
                f'cur_split `{self.cur_split}` not found in data_split, '
                f'will use `train` split to calculate `ctx.var`.')
            cur_split = 'train'
        else:
            cur_split = self.cur_split

        num_batch_last_epoch, num_total_batch = None, None
        if mode in ['train', 'finetune']:
            num_batch, num_batch_last_epoch, num_epoch, num_total_batch = \
                calculate_batch_epoch_num(
                    self.cfg.train.local_update_steps *
                    self.cfg.grad.grad_accum_count,
                    self.cfg.train.batch_or_epoch,
                    self.get(f'num_{cur_split}_data'),
                    self.cfg.dataloader.batch_size,
                    self.cfg.dataloader.drop_last)
        elif mode in ['val', 'test']:
            num_epoch = 1
            num_batch = self.get(f'num_{cur_split}_data'
                                 ) // self.cfg.dataloader.batch_size + int(
                                     not self.cfg.dataloader.drop_last
                                     and bool(
                                         self.get(f'num_{cur_split}_data') %
                                         self.cfg.dataloader.batch_size))
        else:
            raise ValueError(f'Invalid mode {mode}.')

        return num_batch, num_batch_last_epoch, num_epoch, num_total_batch

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
                    f"will skip evaluation."
                    f"If this is not the case you want, please check "
                    f"whether there is typo for the name")
                return False
            else:
                raise ValueError(f"No {target_split_name}_data or"
                                 f" {target_split_name}_loader in the trainer")
        else:
            return True

    def merge_from_dict(self, other_dict):
        for key, value in other_dict.items():
            setattr(self, key, value)


class CtxVar(object):
    """
    Basic variable class

    Arguments:
        lifecycle: specific lifecycle of the attribute
    """

    LIFECYCLES = ["batch", "epoch", "routine", None]

    def __init__(self, obj, lifecycle=None):
        assert lifecycle in CtxVar.LIFECYCLES
        self.obj = obj
        self.lifecycle = lifecycle


def lifecycle(lifecycle):
    """
    Manage the lifecycle of the variables within context, \
    and blind these operations from user.

    Arguments:
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
