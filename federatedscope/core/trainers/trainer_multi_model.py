import copy
from types import FunctionType
from typing import Type

from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer


class GeneralMultiModelTrainer(GeneralTorchTrainer):
    def __init__(self,
                 model_nums,
                 models_interact_mode="sequential",
                 model=None,
                 data=None,
                 device=None,
                 config=None,
                 base_trainer: Type[GeneralTorchTrainer] = None):
        """
            `GeneralMultiModelTrainer` supports train/eval via multiple
            internal models

            Arguments:
                model_nums (int): how many internal models and optimizers
                will be held by the trainer
                models_interact_mode (str): how the models interact, can be
                "sequential" or "parallel".
                model: training model
                data: a dict contains train/val/test data
                device: device to run
                config: for trainer-related configuration
                base_trainer: if given, the GeneralMultiModelTrainer init
                will based on base_trainer copy

                The sequential mode indicates the interaction at
                run_routine level
                [one model runs its whole routine, then do sth. for
                interaction, then next model runs its whole routine]
                ... -> run_routine_model_i
                    -> _switch_model_ctx
                    -> (on_fit_end, _interact_to_other_models)
                    -> run_routine_model_i+1
                    -> ...

                The parallel mode indicates the interaction
                at point-in-time level
                [At a specific point-in-time, one model call hooks (
                including interaction), then next model call hooks]
                ... ->  (on_xxx_point, hook_xxx_model_i)
                    ->  (on_xxx_point, _interact_to_other_models)
                    ->  (on_xxx_point, _switch_model_ctx)
                    ->  (on_xxx_point, hook_xxx_model_i+1)
                    -> ...

        """
        # support two initialization methods for the `GeneralMultiModelTrainer`
        # 1) from another trainer; or 2) standard init manner given (model,
        # data, device, config)
        if base_trainer is None:
            assert model is not None and \
                   data is not None and \
                   device is not None and \
                   config is not None, "when not copy construction, (model, " \
                                       "data, device, config) should not be " \
                                       "None"
            super(GeneralMultiModelTrainer,
                  self).__init__(model, data, device, config)
        else:
            assert isinstance(base_trainer, GeneralMultiModelTrainer) or \
                   issubclass(type(base_trainer), GeneralMultiModelTrainer) \
                   or isinstance(base_trainer, GeneralTorchTrainer) or \
                   issubclass(type(base_trainer), GeneralTorchTrainer) or \
                   "can only copy instances of `GeneralMultiModelTrainer` " \
                   "and its subclasses, or " \
                   "`GeneralTorchTrainer` and its subclasses"
            self.__dict__ = copy.deepcopy(base_trainer.__dict__)

        assert models_interact_mode in ["sequential", "parallel"], \
            f"Invalid models_interact_mode, should be `sequential` or " \
            f"`parallel`, but got {models_interact_mode}"
        self.models_interact_mode = models_interact_mode

        if int(model_nums) != model_nums or model_nums < 1:
            raise ValueError(
                f"model_nums should be integer and >= 1, got {model_nums}.")
        self.model_nums = model_nums

        self.ctx.cur_model_idx = 0  # used to mark cur model

        # different internal models can have different hook_set
        self.hooks_in_train_multiple_models = [self.hooks_in_train]
        self.hooks_in_eval_multiple_models = [self.hooks_in_eval]
        self.init_multiple_models()
        self.init_multiple_model_hooks()
        assert len(self.ctx.models) == model_nums == \
               len(self.hooks_in_train_multiple_models) == len(
            self.hooks_in_eval_multiple_models),\
            "After init, len(hooks_in_train_multiple_models), " \
            "len(hooks_in_eval_multiple_models), " \
            "len(ctx.models) and model_nums should be the same"

    def init_multiple_models(self):
        """
            init multiple models and optimizers: the default implementation
            is copy init manner;
            ========================= Extension =============================
            users can override this function according to their own
            requirements
        """

        additional_models = [
            copy.deepcopy(self.ctx.model) for _ in range(self.model_nums - 1)
        ]
        self.ctx.models = [self.ctx.model] + additional_models

        self.ctx.optimizers = [
            get_optimizer(self.ctx.models[i], **self.cfg.train.optimizer)
            for i in range(0, self.model_nums)
        ]

    def register_multiple_model_hooks(self):
        """
            By default, all internal models adopt the same hook_set.
            ========================= Extension =============================
            Users can override this function to register customized hooks
            for different internal models.

            Note:
                for sequential mode, users can append interact_hook on
                begin/end triggers such as
                    " -> (on_fit_end, _interact_to_other_models) -> "

                for parallel mode, users can append interact_hook on any
                trigger they want such as
                    " -> (on_xxx_point, _interact_to_other_models) -> "

            self.ctx, we must tell the running hooks which data_loader to
            call and which num_samples to count
        """

        self.hooks_in_train_multiple_models.extend([
            self.hooks_in_train_multiple_models[0]
            for _ in range(1, self.model_nums)
        ])
        self.hooks_in_eval_multiple_models.extend([
            self.hooks_in_eval_multiple_models[0]
            for _ in range(1, self.model_nums)
        ])

    def init_multiple_model_hooks(self):
        self.register_multiple_model_hooks()
        if self.models_interact_mode == "sequential":
            # hooks_in_xxx is a list of dict, hooks_in_xxx[i] stores
            # specific set for i-th internal model;
            # for each dict, the key indicates point-in-time and the value
            # indicates specific hook
            self.hooks_in_train = self.hooks_in_train_multiple_models
            self.hooks_in_eval = self.hooks_in_eval_multiple_models
        elif self.models_interact_mode == "parallel":
            # hooks_in_xxx is a dict whose key indicates point-in-time and
            # value indicates specific hook
            for trigger in list(self.hooks_in_train.keys()):
                self.hooks_in_train[trigger] = []
                self.hooks_in_eval[trigger] = []
                for model_idx in range(len(self.ctx.models)):
                    self.hooks_in_train[trigger].extend(
                        self.hooks_in_train_multiple_models[model_idx]
                        [trigger])
                    self.hooks_in_train[trigger].extend(
                        [self._switch_model_ctx])
                    self.hooks_in_eval[trigger].extend(
                        self.hooks_in_eval_multiple_models[model_idx][trigger])
                    self.hooks_in_eval[trigger].extend(
                        [self._switch_model_ctx])
        else:
            raise RuntimeError(
                f"Invalid models_interact_mode, should be `sequential` or "
                f"`parallel`,"
                f" but got {self.models_interact_mode}")

    def register_hook_in_train(self,
                               new_hook,
                               trigger,
                               model_idx=0,
                               insert_pos=None,
                               base_hook=None,
                               insert_mode="before"):
        hooks_dict = self.hooks_in_train_multiple_models[model_idx]
        self._register_hook(base_hook, hooks_dict, insert_mode, insert_pos,
                            new_hook, trigger)

    def register_hook_in_eval(self,
                              new_hook,
                              trigger,
                              model_idx=0,
                              insert_pos=None,
                              base_hook=None,
                              insert_mode="before"):
        hooks_dict = self.hooks_in_eval_multiple_models[model_idx]
        self._register_hook(base_hook, hooks_dict, insert_mode, insert_pos,
                            new_hook, trigger)

    def _switch_model_ctx(self, next_model_idx=None):
        if next_model_idx is None:
            next_model_idx = (self.ctx.cur_model_idx + 1) % len(
                self.ctx.models)
        self.ctx.cur_model_idx = next_model_idx
        self.ctx.model = self.ctx.models[next_model_idx]
        self.ctx.optimizer = self.ctx.optimizers[next_model_idx]

    def _run_routine(self, mode, hooks_set, dataset_name=None):
        """Run the hooks_set and maintain the mode for multiple internal models

        Arguments:
            mode: running mode of client, chosen from train/val/test

        Note:
            Considering evaluation could be in ```hooks_set[
            "on_epoch_end"]```, there could be two data loaders in
        self.ctx, we must tell the running hooks which data_loader to call
        and which num_samples to count

        """
        if self.models_interact_mode == "sequential":
            assert isinstance(hooks_set, list) and isinstance(hooks_set[0],
                                                              dict), \
                "When models_interact_mode=sequential, " \
                "hooks_set should be a list of dict" \
                "hooks_set[i] stores specific set for i-th internal model." \
                "For each dict, the key indicates point-in-time and the " \
                "value indicates specific hook"
            for model_idx in range(len(self.ctx.models)):
                # switch different hooks & ctx for different internal models
                hooks_set_model_i = hooks_set[model_idx]
                self._switch_model_ctx(model_idx)
                # [Interaction at run_routine level]
                # one model runs its whole routine, then do sth. for
                # interaction, then next model runs its whole routine
                # ... -> run_routine_model_i
                #     -> _switch_model_ctx
                #     -> (on_fit_end, _interact_to_other_models)
                #     -> run_routine_model_i+1
                #     -> ...
                super()._run_routine(mode, hooks_set_model_i, dataset_name)
        elif self.models_interact_mode == "parallel":
            assert isinstance(hooks_set, dict), \
                "When models_interact_mode=parallel, hooks_set should be a " \
                "dict whose key indicates point-in-time and value indicates " \
                "specific hook"
            # [Interaction at point-in-time level]
            # at a specific point-in-time, one model call hooks (including
            # interaction), then next model call hooks
            # ... ->  (on_xxx_point, hook_xxx_model_i)
            #     ->  (on_xxx_point, _interact_to_other_models)
            #     ->  (on_xxx_point, _switch_model_ctx)
            #     ->  (on_xxx_point, hook_xxx_model_i+1)
            #     -> ...
            super()._run_routine(mode, hooks_set, dataset_name)
        else:
            raise RuntimeError(
                f"Invalid models_interact_mode, should be `sequential` or "
                f"`parallel`,"
                f" but got {self.models_interact_mode}")

    def get_model_para(self):
        """
            return multiple model parameters
        :return:
        """
        trained_model_para = []
        for model_idx in range(self.model_nums):
            trained_model_para.append(
                self._param_filter(
                    self.ctx.models[model_idx].cpu().state_dict()))

        return trained_model_para[
            0] if self.model_nums == 1 else trained_model_para

    def update(self, model_parameters, strict=False):
        # update multiple model paras
        """
        Arguments:
            model_parameters (list[dict]): Multiple pyTorch Module object's
            state_dict.
        """
        if self.model_nums == 1:
            super().update(model_parameters, strict=strict)
        else:
            assert isinstance(model_parameters, list) and isinstance(
                model_parameters[0], dict), \
                "model_parameters should a list of multiple state_dict"
            assert len(model_parameters) == self.model_nums, \
                f"model_parameters should has the same length to " \
                f"self.model_nums, " \
                f"but got {len(model_parameters)} and {self.model_nums} " \
                f"respectively"
            for model_idx in range(self.model_nums):
                self.ctx.models[model_idx].load_state_dict(self._param_filter(
                    model_parameters[model_idx]),
                                                           strict=strict)

    def train(self, target_data_split_name="train"):
        # return multiple model paras
        sample_size, _, results = super().train(target_data_split_name)

        return sample_size, self.get_model_para(), results
