import collections
import copy
import logging
import os
import numpy as np

from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.auxiliaries.dataloader_builder import WrapDataset
from federatedscope.core.auxiliaries.ReIterator import ReIterator
from federatedscope.core.auxiliaries import utils
from federatedscope.core.trainers.context import Context
from federatedscope.core.monitors.metric_calculator import MetricCalculator

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None
    DataLoader = None
    Dataset = None

logger = logging.getLogger(__name__)


class Trainer(object):
    """
        Register, organize and run the train/test/val procedures
    """

    HOOK_TRIGGER = [
        "on_fit_start", "on_epoch_start", "on_batch_start", "on_batch_forward",
        "on_batch_backward", "on_batch_end", "on_epoch_end", "on_fit_end"
    ]

    def __init__(self, model, data, device, config, only_for_eval=False):
        self.cfg = config
        self.metric_calculator = MetricCalculator(config.eval.metrics)

        self.ctx = Context(model,
                           self.cfg,
                           data,
                           device,
                           init_dict=self.parse_data(data))

        # Atomic operation during training/evaluation
        self.hooks_in_train = collections.defaultdict(list)

        # By default, use the same trigger keys
        self.hooks_in_eval = copy.deepcopy(self.hooks_in_train)

        # register necessary hooks into self.hooks_in_train and self.hooks_in_eval
        if not only_for_eval:
            self.register_default_hooks_train()
        self.register_default_hooks_eval()

        if self.cfg.federate.mode == 'distributed':
            self.print_trainer_meta_info()
        else:
            # in standalone mode, by default, we print the trainer info only once for better logs readability
            pass

    def parse_data(self, data):
        pass

    def register_default_hooks_train(self):
        pass

    def register_default_hooks_eval(self):
        pass

    def reset_hook_in_train(self, target_trigger, target_hook_name=None):
        hooks_dict = self.hooks_in_train
        del_one_hook_idx = self._reset_hook_in_trigger(hooks_dict,
                                                       target_hook_name,
                                                       target_trigger)
        return del_one_hook_idx

    def reset_hook_in_eval(self, target_trigger, target_hook_name=None):
        hooks_dict = self.hooks_in_eval
        del_one_hook_idx = self._reset_hook_in_trigger(hooks_dict,
                                                       target_hook_name,
                                                       target_trigger)
        return del_one_hook_idx

    def replace_hook_in_train(self, new_hook, target_trigger,
                              target_hook_name):
        del_one_hook_idx = self.reset_hook_in_train(
            target_trigger=target_trigger, target_hook_name=target_hook_name)
        self.register_hook_in_train(new_hook=new_hook,
                                    trigger=target_trigger,
                                    insert_pos=del_one_hook_idx)

    def replace_hook_in_eval(self, new_hook, target_trigger, target_hook_name):
        del_one_hook_idx = self.reset_hook_in_eval(
            target_trigger=target_trigger, target_hook_name=target_hook_name)
        self.register_hook_in_eval(new_hook=new_hook,
                                   trigger=target_trigger,
                                   insert_pos=del_one_hook_idx)

    def _reset_hook_in_trigger(self, hooks_dict, target_hook_name,
                               target_trigger):
        # clean/delete existing hooks for a specific trigger,
        # if target_hook_name given, will clean only the specific one; otherwise, will clean all hooks for the trigger.
        assert target_trigger in self.HOOK_TRIGGER, \
            f"Got {target_trigger} as hook trigger, you should specify a string within {self.HOOK_TRIGGER}."
        del_one_hook_idx = None
        if target_hook_name is None:
            hooks_dict[target_trigger] = []
            del_one_hook_idx = -1  # -1 indicates del the whole list
        else:
            for hook_idx in range(len(hooks_dict[target_trigger])):
                if target_hook_name == hooks_dict[target_trigger][
                        hook_idx].__name__:
                    del_one = hooks_dict[target_trigger].pop(hook_idx)
                    logger.info(
                        f"Remove the hook `{del_one}` from hooks_set at trigger `{target_trigger}`"
                    )
                    del_one_hook_idx = hook_idx
                    break
            if del_one_hook_idx is None:
                logger.warning(
                    f"In hook del procedure, can't find the target hook named {target_hook_name}"
                )
        return del_one_hook_idx

    def register_hook_in_train(self,
                               new_hook,
                               trigger,
                               insert_pos=None,
                               base_hook=None,
                               insert_mode="before"):
        hooks_dict = self.hooks_in_train
        self._register_hook(base_hook, hooks_dict, insert_mode, insert_pos,
                            new_hook, trigger)

    def register_hook_in_eval(self,
                              new_hook,
                              trigger,
                              insert_pos=None,
                              base_hook=None,
                              insert_mode="before"):
        hooks_dict = self.hooks_in_eval
        self._register_hook(base_hook, hooks_dict, insert_mode, insert_pos,
                            new_hook, trigger)

    def _register_hook(self, base_hook, hooks_dict, insert_mode, insert_pos,
                       new_hook, trigger):
        assert trigger in self.HOOK_TRIGGER, \
            f"Got {trigger} as hook trigger, you should specify a string within {self.HOOK_TRIGGER}."
        # parse the insertion position
        target_hook_set = hooks_dict[trigger]
        if insert_pos is not None:
            assert (insert_pos == -1) or (insert_pos == len(target_hook_set) == 0) or \
                   (0 <= insert_pos <= (len(target_hook_set) - 1)), \
                f"Got {insert_pos} as insert pos, you should specify a integer (1) =-1 " \
                f"or (2) =0 for null target_hook_set;" \
                f"or (3) within [0, {(len(target_hook_set) - 1)}]."
        elif base_hook is not None:
            base_hook_pos = target_hook_set.index(base_hook)
            insert_pos = base_hook_pos - 1 if insert_mode == "before" else base_hook_pos + 1
            # bounding the insert_pos in rational range
            insert_pos = 0 if insert_pos < 0 else insert_pos
            insert_pos = -1 if insert_pos >= len(
                target_hook_set) else insert_pos
        else:
            insert_pos = -1  # By default, the new hook is called finally
        # register the new hook
        if insert_pos == -1:
            hooks_dict[trigger].append(new_hook)
        else:
            hooks_dict[trigger].insert(insert_pos, new_hook)

    def train(self, target_data_split_name="train", hooks_set=None):
        pass

    def evaluate(self, target_data_split_name="test", hooks_set=None):
        hooks_set = self.hooks_in_eval if hooks_set is None else hooks_set
        if self.ctx.get(
                f"{target_data_split_name}_data") is None and self.ctx.get(
                    f"{target_data_split_name}_loader") is None:
            logger.warning(
                f"No {target_data_split_name}_data or {target_data_split_name}_loader in the trainer, will skip evaluation"
                f"If this is not the case you want, please check whether there is typo for the name"
            )
            self.ctx.eval_metrics = {}
        else:
            self._run_routine("test", hooks_set, target_data_split_name)

        return self.ctx.eval_metrics

    def _run_routine(self, mode, hooks_set, dataset_name=None):
        """Run the hooks_set and maintain the mode

        Arguments:
            mode: running mode of client, chosen from train/val/test

        Note:
            Considering evaluation could be in ```hooks_set["on_epoch_end"]```, there could be two data loaders in
        self.ctx, we must tell the running hooks which data_loader to call and which num_samples to count

        """
        if dataset_name is None:
            dataset_name = mode
        self.ctx.append_mode(mode)
        self.ctx.track_used_dataset(dataset_name)

        for hook in hooks_set["on_fit_start"]:
            hook(self.ctx)

        for epoch_i in range(self.ctx.get(
                "num_{}_epoch".format(dataset_name))):
            self.ctx.cur_epoch_i = epoch_i
            for hook in hooks_set["on_epoch_start"]:
                hook(self.ctx)

            for batch_i in range(
                    self.ctx.get("num_{}_batch".format(dataset_name))):
                self.ctx.cur_batch_i = batch_i
                for hook in hooks_set["on_batch_start"]:
                    hook(self.ctx)
                for hook in hooks_set["on_batch_forward"]:
                    hook(self.ctx)
                if self.ctx.cur_mode == 'train':
                    for hook in hooks_set["on_batch_backward"]:
                        hook(self.ctx)
                for hook in hooks_set["on_batch_end"]:
                    hook(self.ctx)

                # Break in the final epoch
                if self.ctx.cur_mode == 'train' and epoch_i == self.ctx.num_train_epoch - 1:
                    if batch_i >= self.ctx.num_train_batch_last_epoch - 1:
                        break

            for hook in hooks_set["on_epoch_end"]:
                hook(self.ctx)
        for hook in hooks_set["on_fit_end"]:
            hook(self.ctx)

        self.ctx.pop_mode()
        self.ctx.reset_used_dataset()
        # Avoid memory leak
        if not self.cfg.federate.share_local_model:
            if torch is None:
                pass
            else:
                self.ctx.model.to(torch.device("cpu"))

    def update(self, model_parameters):
        '''
            Called by the FL client to update the model parameters
        Arguments:
            model_parameters (dict): {model_name: model_val}
        '''
        pass

    def get_model_para(self):
        '''

        :return: model_parameters (dict): {model_name: model_val}
        '''
        pass

    def print_trainer_meta_info(self):
        '''
            print some meta info for code-users, e.g., model type; the para names will be filtered out, etc.,
        '''
        logger.info(f"Model meta-info: {type(self.ctx.model)}.")
        logger.debug(f"Model meta-info: {self.ctx.model}.")
        # logger.info(f"Data meta-info: {self.ctx['data']}.")

        ori_para_names = set(self.ctx.model.state_dict().keys())
        preserved_paras = self._param_filter(self.ctx.model.state_dict())
        preserved_para_names = set(preserved_paras.keys())
        filtered_para_names = ori_para_names - preserved_para_names
        logger.info(f"Num of original para names: {len(ori_para_names)}.")
        logger.info(
            f"Num of original trainable para names: {len(self.ctx['trainable_para_names'])}."
        )
        logger.info(
            f"Num of preserved para names in local update: {len(preserved_para_names)}. \n"
            f"Preserved para names in local update: {preserved_para_names}.")
        logger.info(
            f"Num of filtered para names in local update: {len(filtered_para_names)}. \n"
            f"Filtered para names in local update: {filtered_para_names}.")

        logger.info(f"After register default hooks,\n"
                    f"\tthe hooks_in_train is: {self.hooks_in_train};\n"
                    f"\tthe hooks_in_eval is {self.hooks_in_eval}")

    def finetune(self):
        pass

    def _param_filter(self, state_dict, filter_keywords=None):
        '''
        model parameter filter when transmit between local and gloabl, which is useful in personalization.
        e.g., setting cfg.personalization.local_param= ['bn', 'norms'] indicates the implementation of
        "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization, ICML2021", which can be found in https://openreview.net/forum?id=6YEQUn0QICG

        Arguments:
            state_dict (dict): PyTorch Module object's state_dict.
        Returns:
            state_dict (dict): remove the keys that match any of the given keywords.
        '''
        if self.cfg.federate.method in ["local", "global"]:
            return {}

        if filter_keywords is None:
            filter_keywords = self.cfg.personalization.local_param

        trainable_filter = lambda p: True if self.cfg.personalization.share_non_trainable_para else \
            lambda p: p in self.ctx.trainable_para_names
        keyword_filter = utils.filter_by_specified_keywords
        return dict(
            filter(
                lambda elem: trainable_filter(elem[1]) and keyword_filter(
                    elem[0], filter_keywords), state_dict.items()))

    def save_model(self, path, cur_round=-1):
        raise NotImplementedError(
            "The function `save_model` should be implemented according to the ML backend (Pytorch, Tensorflow ...)."
        )

    def load_model(self, path):
        raise NotImplementedError(
            "The function `load_model` should be implemented according to the ML backend (Pytorch, Tensorflow ...)."
        )


class GeneralTorchTrainer(Trainer):
    def get_model_para(self):
        return self._param_filter(
            self.ctx.model.state_dict() if self.cfg.federate.
            share_local_model else self.ctx.model.cpu().state_dict())

    def parse_data(self, data):
        """Populate "{}_data", "{}_loader" and "num_{}_data" for different modes

        """
        # TODO: more robust for different data
        init_dict = dict()
        if isinstance(data, dict):
            for mode in ["train", "val", "test"]:
                init_dict["{}_data".format(mode)] = None
                init_dict["{}_loader".format(mode)] = None
                init_dict["num_{}_data".format(mode)] = 0
                if data.get(mode, None) is not None:
                    if isinstance(data.get(mode), Dataset):
                        init_dict["{}_data".format(mode)] = data.get(mode)
                        init_dict["num_{}_data".format(mode)] = len(
                            data.get(mode))
                    elif isinstance(data.get(mode), DataLoader):
                        init_dict["{}_loader".format(mode)] = data.get(mode)
                        init_dict["num_{}_data".format(mode)] = len(
                            data.get(mode).dataset)
                    elif isinstance(data.get(mode), dict):
                        init_dict["{}_data".format(mode)] = data.get(mode)
                        init_dict["num_{}_data".format(mode)] = len(
                            data.get(mode)['y'])
                    else:
                        raise TypeError("Type {} is not supported.".format(
                            type(data.get(mode))))
        else:
            raise TypeError("Type of data should be dict.")
        return init_dict

    def train(self, target_data_split_name="train", hooks_set=None):
        hooks_set = self.hooks_in_train if hooks_set is None else hooks_set
        if self.ctx.get(
                f"{target_data_split_name}_data") is None and self.ctx.get(
                    f"{target_data_split_name}_loader") is None:
            raise ValueError(
                f"No {target_data_split_name}_data or {target_data_split_name}_loader in the trainer"
            )
        self._run_routine("train", hooks_set, target_data_split_name)

        # TODO: The return values should be more flexible? Now: sample_num, model_para, results={k:v}

        return self.ctx.num_samples_train, self.get_model_para(
        ), self.ctx.eval_metrics

    def update(self, model_parameters):
        '''
            Called by the FL client to update the model parameters
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        '''
        for key in model_parameters:
            if isinstance(model_parameters[key], list):
                model_parameters[key] = torch.FloatTensor(
                    model_parameters[key])
        self.ctx.model.load_state_dict(self._param_filter(model_parameters),
                                       strict=False)

    def evaluate(self, target_data_split_name="test"):
        with torch.no_grad():
            super().evaluate(target_data_split_name)

        return self.ctx.eval_metrics

    def validate(self, target_data_split_name="val"):
        with torch.no_grad():
            super().evaluate(target_data_split_name)

        return self.ctx.eval_metrics

    def finetune(self, target_data_split_name="train", hooks_set=None):

        # freeze the parameters during the fine-tune stage
        require_grad_changed_paras = set()
        if self.cfg.trainer.finetune.freeze_param != "":
            preserved_paras = self._param_filter(self.ctx.model.state_dict(), self.cfg.trainer.finetune.freeze_param)
            for name, param in self.ctx.model.named_parameters():
                if name not in preserved_paras and param.requires_grad is True:
                    param.requires_grad = False
                    require_grad_changed_paras.add(name)

        # change the optimization configs
        original_lrs = []
        for g in self.ctx.optimizer.param_groups:
            original_lrs.append(g['lr'])
            g['lr'] = self.cfg.trainer.finetune.lr
        original_epoch_num = self.ctx["num_train_epoch"]
        original_batch_num = self.ctx["num_train_batch"]
        self.ctx["num_train_epoch"] = 1
        self.ctx["num_train_batch"] = self.cfg.trainer.finetune.steps

        # do the fine-tuning process
        self.train(target_data_split_name, hooks_set)

        # restore the state before fine-tuning
        if len(require_grad_changed_paras) > 0:
            for name, param in self.ctx.model.named_parameters():
                if name in require_grad_changed_paras:
                    param.requires_grad = True

        for i, g in enumerate(self.ctx.optimizer.param_groups):
            g['lr'] = original_lrs[i]

        self.ctx["num_train_epoch"] = original_epoch_num
        self.ctx["num_train_batch"] = original_batch_num

    def register_default_hooks_train(self):
        self.register_hook_in_train(self._hook_on_fit_start_init,
                                    "on_fit_start")
        self.register_hook_in_train(self._hook_on_epoch_start,
                                    "on_epoch_start")
        self.register_hook_in_train(self._hook_on_batch_start_init,
                                    "on_batch_start")
        self.register_hook_in_train(self._hook_on_batch_forward,
                                    "on_batch_forward")
        self.register_hook_in_train(self._hook_on_batch_forward_regularizer,
                                    "on_batch_forward")
        self.register_hook_in_train(self._hook_on_batch_backward,
                                    "on_batch_backward")
        self.register_hook_in_train(self._hook_on_batch_end, "on_batch_end")
        self.register_hook_in_train(self._hook_on_fit_end, "on_fit_end")

    def register_default_hooks_eval(self):
        # test/val
        self.register_hook_in_eval(self._hook_on_fit_start_init,
                                   "on_fit_start")
        self.register_hook_in_eval(self._hook_on_epoch_start, "on_epoch_start")
        self.register_hook_in_eval(self._hook_on_batch_start_init,
                                   "on_batch_start")
        self.register_hook_in_eval(self._hook_on_batch_forward,
                                   "on_batch_forward")
        self.register_hook_in_eval(self._hook_on_batch_end, "on_batch_end")
        self.register_hook_in_eval(self._hook_on_fit_end, "on_fit_end")

    def _hook_on_fit_start_init(self, ctx):
        # prepare model
        ctx.model.to(ctx.device)

        # prepare statistics
        setattr(ctx, "loss_batch_total_{}".format(ctx.cur_data_split), 0)
        setattr(ctx, "loss_regular_total_{}".format(ctx.cur_data_split), 0)
        setattr(ctx, "num_samples_{}".format(ctx.cur_data_split), 0)
        setattr(ctx, "{}_y_true".format(ctx.cur_data_split), [])
        setattr(ctx, "{}_y_prob".format(ctx.cur_data_split), [])

    def _hook_on_epoch_start(self, ctx):
        # prepare dataloader
        if ctx.get("{}_loader".format(ctx.cur_data_split)) is None:
            loader = get_dataloader(
                WrapDataset(ctx.get("{}_data".format(ctx.cur_data_split))),
                self.cfg)
            setattr(ctx, "{}_loader".format(ctx.cur_data_split),
                    ReIterator(loader))
        elif not isinstance(ctx.get("{}_loader".format(ctx.cur_data_split)),
                            ReIterator):
            setattr(
                ctx, "{}_loader".format(ctx.cur_data_split),
                ReIterator(ctx.get("{}_loader".format(ctx.cur_data_split))))
        else:
            ctx.get("{}_loader".format(ctx.cur_data_split)).reset()

    def _hook_on_batch_start_init(self, ctx):
        # prepare data batch
        try:
            ctx.data_batch = next(
                ctx.get("{}_loader".format(ctx.cur_data_split)))
        except StopIteration:
            raise StopIteration

    def _hook_on_batch_forward(self, ctx):
        x, label = [_.to(ctx.device) for _ in ctx.data_batch]
        pred = ctx.model(x)
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
        ctx.loss_batch = ctx.criterion(pred, label)
        ctx.y_true = label
        ctx.y_prob = pred

        ctx.batch_size = len(label)

    def _hook_on_batch_forward_regularizer(self, ctx):
        ctx.loss_regular = float(
            self.cfg.regularizer.mu) * ctx.regularizer(ctx)
        ctx.loss_task = ctx.loss_batch + ctx.loss_regular

    def _hook_on_batch_backward(self, ctx):
        ctx.optimizer.zero_grad()
        ctx.loss_task.backward()
        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
        ctx.optimizer.step()

    def _hook_on_batch_end(self, ctx):
        # update statistics
        setattr(
            ctx, "loss_batch_total_{}".format(ctx.cur_data_split),
            ctx.get("loss_batch_total_{}".format(ctx.cur_data_split)) +
            ctx.loss_batch.item() * ctx.batch_size)

        if ctx.get("loss_regular", None) is None or ctx.loss_regular == 0:
            loss_regular = 0.
        else:
            loss_regular = ctx.loss_regular.item()
        setattr(
            ctx, "loss_regular_total_{}".format(ctx.cur_data_split),
            ctx.get("loss_regular_total_{}".format(ctx.cur_data_split)) +
            loss_regular)
        setattr(
            ctx, "num_samples_{}".format(ctx.cur_data_split),
            ctx.get("num_samples_{}".format(ctx.cur_data_split)) +
            ctx.batch_size)

        # cache label for evaluate
        ctx.get("{}_y_true".format(ctx.cur_data_split)).append(
            ctx.y_true.detach().cpu().numpy())

        ctx.get("{}_y_prob".format(ctx.cur_data_split)).append(
            ctx.y_prob.detach().cpu().numpy())

        # clean temp ctx
        ctx.data_batch = None
        ctx.batch_size = None
        ctx.loss_task = None
        ctx.loss_batch = None
        ctx.loss_regular = None
        ctx.y_true = None
        ctx.y_prob = None

    def _hook_on_fit_end(self, ctx):
        """Evaluate metrics.

        """
        setattr(
            ctx, "{}_y_true".format(ctx.cur_data_split),
            np.concatenate(ctx.get("{}_y_true".format(ctx.cur_data_split))))
        setattr(
            ctx, "{}_y_prob".format(ctx.cur_data_split),
            np.concatenate(ctx.get("{}_y_prob".format(ctx.cur_data_split))))
        results = self.metric_calculator.eval(ctx)
        setattr(ctx, 'eval_metrics', results)

    def save_model(self, path, cur_round=-1):
        assert self.ctx.model is not None

        ckpt = {'cur_round': cur_round, 'model': self.ctx.model.state_dict()}
        torch.save(ckpt, path)

    def load_model(self, path):
        assert self.ctx.model is not None

        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.ctx.device)
            self.ctx.model.load_state_dict(ckpt['model'])
            return ckpt['cur_round']
        else:
            raise ValueError("The file {} does NOT exist".format(path))
