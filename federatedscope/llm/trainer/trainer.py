from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE
from federatedscope.core.auxiliaries.utils import param2tensor, \
    merge_param_dict


class LLMTrainer(GeneralTorchTrainer):
    def update(self, model_parameters, strict=False):
        # TODO: enable adapter
        """
            Called by the FL client to update the model parameters
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        """
        for key in model_parameters:
            model_parameters[key] = param2tensor(model_parameters[key])
        # Due to lazy load, we merge two state dict
        merged_param = merge_param_dict(self.ctx.model.state_dict().copy(),
                                        self._param_filter(model_parameters))
        self.ctx.model.load_state_dict(merged_param, strict=strict)

    def _hook_on_batch_forward(self, ctx):
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)

        outputs = ctx.model.forward(input_ids, labels=labels)

        logits = outputs.logits
        loss = outputs.loss

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_fit_end(self, ctx):
        # TODO: enable other metrics in
        #  https://crfm-helm.readthedocs.io/en/latest/metrics/
        eval_results = {
            f'{ctx.cur_split}_loss': ctx.loss_batch_total,
            f'{ctx.cur_split}_total': ctx.num_samples,
            f'{ctx.cur_split}_avg_loss': ctx.loss_batch_total /
            float(ctx.num_samples),
        }
        setattr(ctx, 'eval_metrics', eval_results)


def call_llm_trainer(trainer_type):
    if trainer_type == 'llmtrainer':
        trainer_builder = LLMTrainer
        return trainer_builder


register_trainer('llmtrainer', call_llm_trainer)
