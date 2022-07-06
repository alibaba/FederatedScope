from typing import Type

import numpy as np
import torch
from torch.nn.functional import softmax as f_softmax

from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.trainers.trainer_multi_model import \
    GeneralMultiModelTrainer


class FedEMTrainer(GeneralMultiModelTrainer):
    """
    The FedEM implementation, "Federated Multi-Task Learning under a
    Mixture of Distributions (NeurIPS 2021)"
    based on the Algorithm 1 in their paper and official codes:
    https://github.com/omarfoq/FedEM
    """
    def __init__(self,
                 model_nums,
                 models_interact_mode="sequential",
                 model=None,
                 data=None,
                 device=None,
                 config=None,
                 base_trainer: Type[GeneralTorchTrainer] = None):
        super(FedEMTrainer,
              self).__init__(model_nums, models_interact_mode, model, data,
                             device, config, base_trainer)
        device = self.ctx.device

        # --------------- attribute-level modifications ----------------------
        # used to mixture the internal models
        self.weights_internal_models = (torch.ones(self.model_nums) /
                                        self.model_nums).to(device)
        self.weights_data_sample = (
            torch.ones(self.model_nums, self.ctx.num_train_batch) /
            self.model_nums).to(device)

        self.ctx.all_losses_model_batch = torch.zeros(
            self.model_nums, self.ctx.num_train_batch).to(device)
        self.ctx.cur_batch_idx = -1
        # `ctx[f"{cur_data}_y_prob_ensemble"] = 0` in
        #   func `_hook_on_fit_end_ensemble_eval`
        #   -> self.ctx.test_y_prob_ensemble = 0
        #   -> self.ctx.train_y_prob_ensemble = 0
        #   -> self.ctx.val_y_prob_ensemble = 0

        # ---------------- action-level modifications -----------------------
        # see register_multiple_model_hooks(),
        # which is called in the __init__ of `GeneralMultiModelTrainer`

    def register_multiple_model_hooks(self):
        """
            customized multiple_model_hooks, which is called
            in the __init__ of `GeneralMultiModelTrainer`
        """
        # First register hooks for model 0
        # ---------------- train hooks -----------------------
        self.register_hook_in_train(
            new_hook=self.hook_on_fit_start_mixture_weights_update,
            trigger="on_fit_start",
            insert_pos=0)  # insert at the front
        self.register_hook_in_train(
            new_hook=self._hook_on_fit_start_flop_count,
            trigger="on_fit_start",
            insert_pos=1  # follow the mixture operation
        )
        self.register_hook_in_train(new_hook=self._hook_on_fit_end_flop_count,
                                    trigger="on_fit_end",
                                    insert_pos=-1)
        self.register_hook_in_train(
            new_hook=self.hook_on_batch_forward_weighted_loss,
            trigger="on_batch_forward",
            insert_pos=-1)
        self.register_hook_in_train(
            new_hook=self.hook_on_batch_start_track_batch_idx,
            trigger="on_batch_start",
            insert_pos=0)  # insert at the front
        # ---------------- eval hooks -----------------------
        self.register_hook_in_eval(
            new_hook=self.hook_on_batch_end_gather_loss,
            trigger="on_batch_end",
            insert_pos=0
        )  # insert at the front, (we need gather the loss before clean it)
        self.register_hook_in_eval(
            new_hook=self.hook_on_batch_start_track_batch_idx,
            trigger="on_batch_start",
            insert_pos=0)  # insert at the front
        # replace the original evaluation into the ensemble one
        self.replace_hook_in_eval(new_hook=self._hook_on_fit_end_ensemble_eval,
                                  target_trigger="on_fit_end",
                                  target_hook_name="_hook_on_fit_end")

        # Then for other models, set the same hooks as model 0
        # since we differentiate different models in the hook
        # implementations via ctx.cur_model_idx
        self.hooks_in_train_multiple_models.extend([
            self.hooks_in_train_multiple_models[0]
            for _ in range(1, self.model_nums)
        ])
        self.hooks_in_eval_multiple_models.extend([
            self.hooks_in_eval_multiple_models[0]
            for _ in range(1, self.model_nums)
        ])

    def hook_on_batch_start_track_batch_idx(self, ctx):
        # for both train & eval
        ctx.cur_batch_idx = (self.ctx.cur_batch_idx +
                             1) % self.ctx.num_train_batch

    def hook_on_batch_forward_weighted_loss(self, ctx):
        # for only train
        ctx.loss_batch *= self.weights_internal_models[ctx.cur_model_idx]

    def hook_on_batch_end_gather_loss(self, ctx):
        # for only eval
        # before clean the loss_batch; we record it
        # for further weights_data_sample update
        ctx.all_losses_model_batch[ctx.cur_model_idx][
            ctx.cur_batch_idx] = ctx.loss_batch.item()

    def hook_on_fit_start_mixture_weights_update(self, ctx):
        # for only train
        if ctx.cur_model_idx != 0:
            # do the mixture_weights_update once
            pass
        else:
            # gathers losses for all sample in iterator
            # for each internal model, calling *evaluate()*
            for model_idx in range(self.model_nums):
                self._switch_model_ctx(model_idx)
                self.evaluate(target_data_split_name="train")

            self.weights_data_sample = f_softmax(
                (torch.log(self.weights_internal_models) -
                 ctx.all_losses_model_batch.T),
                dim=1).T
            self.weights_internal_models = self.weights_data_sample.mean(dim=1)

            # restore the model_ctx
            self._switch_model_ctx(0)

    def _hook_on_fit_start_flop_count(self, ctx):
        self.ctx.monitor.total_flops += self.ctx.monitor.flops_per_sample * \
                                        self.model_nums * ctx.num_train_data

    def _hook_on_fit_end_flop_count(self, ctx):
        self.ctx.monitor.total_flops += self.ctx.monitor.flops_per_sample * \
                                        self.model_nums * ctx.num_train_data

    def _hook_on_fit_end_ensemble_eval(self, ctx):
        """
            Ensemble evaluation
        """
        cur_data = ctx.cur_data_split
        if f"{cur_data}_y_prob_ensemble" not in ctx:
            ctx[f"{cur_data}_y_prob_ensemble"] = 0
        ctx[f"{cur_data}_y_prob_ensemble"] += \
            np.concatenate(ctx[f"{cur_data}_y_prob"]) * \
            self.weights_internal_models[ctx.cur_model_idx].item()

        # do metrics calculation after the last internal model evaluation done
        if ctx.cur_model_idx == self.model_nums - 1:
            ctx[f"{cur_data}_y_true"] = np.concatenate(
                ctx[f"{cur_data}_y_true"])
            ctx[f"{cur_data}_y_prob"] = ctx[f"{cur_data}_y_prob_ensemble"]
            ctx.eval_metrics = self.metric_calculator.eval(ctx)
            # reset for next run_routine that may have different len([f"{
            # cur_data}_y_prob"])
            ctx[f"{cur_data}_y_prob_ensemble"] = 0

        ctx[f"{cur_data}_y_prob"] = []
        ctx[f"{cur_data}_y_true"] = []
