import logging

import math

from federatedscope.core.auxiliaries.criterion_builder import get_criterion
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.model_builder import \
    get_trainable_para_names
from federatedscope.core.auxiliaries.regularizer_builder import get_regularizer


class Context(dict):
    """Record and pass variables among different hook functions.

    Arguments:
        model (Module): training model
        data (dict): a dict contains train/val/test dataset or dataloader
        device: running device

    Record attributes:
        - model (Module): the training model
        - data (dict): a dict contains train/val/test dataset or dataloader
        - device (torch.device): specific device to running to
        - criterion: specific loss function
        - optimizer: specific optimizer
        - mode: maintain the current mode of the model

        - data_batch: current batch data from train/test/val data loader

        - trainable_para_names (list): a list of the names of the trainable
        parameters within ```ctx.model```
        - train_data: training dataset
        - train_loader: training dataloader
        - num_train_data (int): the number of training samples within one epoch
        - num_train_epoch (int): the number of total training epochs
        - num_train_batch (int): the number of batches within one completed
        training epoch
        - num_train_batch_last_epoch (int): the number of batches within
        the last epoch

        - test_data: test data
        - test_loader: test dataloader
        - num_test_data (int): the number of test samples within one epoch
        - num_test_epoch (int): the number of test epochs, default 1
        - num_test_batch (int): the number of batches within one completed
        test epoch

        - val_data: val data
        - val_loader: val dataloader
        - num_val_data (int): the number of val samples within one epoch
        - num_val_epoch (int): the number of val epochs, default 1
        - num_val_batch (int): the number of batches within one completed
        val epoch

    Statistical variables:
        - loss_batch (float): loss of the current data_batch, shared by
        train/test/val
        - loss_regular (float): loss of the regularizer
        - loss_task (float): the sum of loss_batch and loss_regular

        - loss_total_batch_train (float): accumulated batch loss during
        training
        - loss_total_regular_train (float): accumulated regular loss during
        training
        - num_samples_train (int): accumulated number of training samples
        involved at present

        - loss_total_test (float): accumulated batch loss during test
        - num_samples_test (float): accumulated regular loss during test

        - loss_total_val (float): accumulated batch loss during val
        - num_samples_val (float): accumulated regular loss during val

        - eval_metrics (dict): evaluation results
    """

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError("Attribute {} is not found".format(item))

    def __init__(self,
                 model,
                 cfg,
                 data=None,
                 device=None,
                 init_dict=None,
                 init_attr=True):
        if init_dict is None:
            super(Context, self).__init__()
        else:
            super(Context, self).__init__(init_dict)

        self.cfg = cfg
        self.model = model
        self.data = data
        self.device = device
        self.cur_mode = None
        self.cur_data_split = None

        if init_attr:
            # setup static variables for training/evaluation
            self.setup_vars()

    def setup_vars(self):
        if self.cfg.backend == 'torch':
            self.trainable_para_names = get_trainable_para_names(self.model)
            self.criterion = get_criterion(self.cfg.criterion.type,
                                           self.device)
            self.regularizer = get_regularizer(self.cfg.regularizer.type)
            self.optimizer = get_optimizer(self.model, **self.cfg.optimizer)
            self.grad_clip = self.cfg.grad.grad_clip
        elif self.cfg.backend == 'tensorflow':
            self.trainable_para_names = self.model.trainable_variables()
            self.criterion = None
            self.regularizer = None
            self.optimizer = None
            self.grad_clip = None

        self.mode = list()
        self.cur_data_splits_used_by_routine = list()

        # Process training data
        if self.train_data is not None or self.train_loader is not None:
            # Calculate the number of update steps during training given the
            # local_update_steps
            num_train_batch, num_train_batch_last_epoch, num_train_epoch, \
                num_total_train_batch = self.pre_calculate_batch_epoch_num(
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

    def pre_calculate_batch_epoch_num(self, local_update_steps):
        num_train_batch = self.num_train_data // self.cfg.data.batch_size + \
                          int(not self.cfg.data.drop_last and bool(
                              self.num_train_data % self.cfg.data.batch_size))
        if self.cfg.federate.batch_or_epoch == "epoch":
            num_train_epoch = local_update_steps
            num_train_batch_last_epoch = num_train_batch
            num_total_train_batch = local_update_steps * num_train_batch
        elif num_train_batch == 0:
            raise RuntimeError(
                "The number of training batch is 0, please check "
                "'batch_size' or set 'drop_last' as False")
        else:
            num_train_epoch = math.ceil(local_update_steps / num_train_batch)
            num_train_batch_last_epoch = local_update_steps % \
                num_train_batch or num_train_batch
            num_total_train_batch = local_update_steps
        return num_train_batch, num_train_batch_last_epoch, num_train_epoch,\
            num_total_train_batch

    def append_mode(self, mode):
        self.mode.append(mode)
        self.cur_mode = self.mode[-1]
        self.change_mode(self.cur_mode)

    def pop_mode(self):
        self.mode.pop()
        self.cur_mode = self.mode[-1] if len(self.mode) != 0 else None
        if len(self.mode) != 0:
            self.change_mode(self.cur_mode)

    def change_mode(self, mode):
        # change state
        if self.cfg.backend == 'torch':
            getattr(self.model, mode if mode == 'train' else 'eval')()
        else:
            pass

    def track_used_dataset(self, dataset):
        # stack-style to enable mixture usage such as evaluation on train
        # dataset
        self.cur_data_splits_used_by_routine.append(dataset)
        self.cur_data_split = self.cur_data_splits_used_by_routine[-1]

    def reset_used_dataset(self):
        self.cur_data_splits_used_by_routine.pop()
        self.cur_data_split = self.cur_data_splits_used_by_routine[-1] if \
            len(self.cur_data_splits_used_by_routine) != 0 else None
