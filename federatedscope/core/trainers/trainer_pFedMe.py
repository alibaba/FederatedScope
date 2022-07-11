import copy

from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.optimizer import wrap_regularized_optimizer
from typing import Type


def wrap_pFedMeTrainer(
        base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:
    """
    Build a `pFedMeTrainer` with a plug-in manner, by registering new
    functions into specific `BaseTrainer`

    The pFedMe implementation, "Personalized Federated Learning with Moreau
    Envelopes (NeurIPS 2020)"
    is based on the Algorithm 1 in their paper and official codes:
    https://github.com/CharlieDinh/pFedMe
    """

    # ---------------- attribute-level plug-in -----------------------
    init_pFedMe_ctx(base_trainer)

    # ---------------- action-level plug-in -----------------------
    base_trainer.register_hook_in_train(
        new_hook=hook_on_fit_start_set_local_para_tmp,
        trigger="on_fit_start",
        insert_pos=-1)
    base_trainer.register_hook_in_train(
        new_hook=hook_on_epoch_end_update_local,
        trigger="on_epoch_end",
        insert_pos=-1)
    base_trainer.register_hook_in_train(new_hook=hook_on_fit_end_update_local,
                                        trigger="on_fit_end",
                                        insert_pos=-1)

    base_trainer.register_hook_in_train(new_hook=_hook_on_batch_end_flop_count,
                                        trigger="on_batch_end",
                                        insert_pos=-1)
    base_trainer.register_hook_in_train(new_hook=_hook_on_epoch_end_flop_count,
                                        trigger="on_epoch_end",
                                        insert_pos=-1)

    # for "on_batch_start" trigger: replace the original hooks into new ones
    # of pFedMe
    # 1) cache the original hooks for "on_batch_start"
    base_trainer.ctx.original_hook_on_batch_start_train = \
        base_trainer.hooks_in_train["on_batch_start"]
    base_trainer.ctx.original_hook_on_batch_start_eval = \
        base_trainer.hooks_in_eval["on_batch_start"]
    # 2) replace the original hooks for "on_batch_start"
    base_trainer.replace_hook_in_train(
        new_hook=hook_on_batch_start_init_pfedme,
        target_trigger="on_batch_start",
        target_hook_name=None)
    base_trainer.replace_hook_in_eval(new_hook=hook_on_batch_start_init_pfedme,
                                      target_trigger="on_batch_start",
                                      target_hook_name=None)

    return base_trainer


def init_pFedMe_ctx(base_trainer):
    """
    init necessary attributes used in pFedMe,
    some new attributes will be with prefix `pFedMe` optimizer to avoid
    namespace pollution
    """
    ctx = base_trainer.ctx
    cfg = base_trainer.cfg

    # pFedMe finds approximate model with K steps using the same data batch
    # the complexity of each pFedMe client is K times the one of FedAvg
    ctx.pFedMe_K = cfg.personalization.K
    ctx.num_train_epoch *= ctx.pFedMe_K
    ctx.pFedMe_approx_fit_counter = 0

    # the local_model_tmp is used to be the referenced parameter when
    # finding the approximate \theta in paper
    # will be copied from model every run_routine
    ctx.pFedMe_local_model_tmp = None


def hook_on_fit_start_set_local_para_tmp(ctx):
    # the optimizer used in pFedMe is based on Moreau Envelopes regularization
    # besides, there are two distinct lr for the approximate model and base
    # model
    ctx.optimizer = wrap_regularized_optimizer(
        ctx.optimizer, ctx.cfg.personalization.regular_weight)
    for g in ctx.optimizer.param_groups:
        g['lr'] = ctx.cfg.personalization.lr
    ctx.pFedMe_outer_lr = ctx.cfg.train.optimizer.lr

    ctx.pFedMe_local_model_tmp = copy.deepcopy(ctx.model)
    # set the compared model data, then the optimizer will find approximate
    # model using trainer.cfg.personalization.lr
    compared_global_model_para = [{
        "params": list(ctx.pFedMe_local_model_tmp.parameters())
    }]
    ctx.optimizer.set_compared_para_group(compared_global_model_para)


def hook_on_batch_start_init_pfedme(ctx):
    # refresh data every K step
    if ctx.pFedMe_approx_fit_counter == 0:
        if ctx.cur_mode == "train":
            for hook in ctx.original_hook_on_batch_start_train:
                hook(ctx)
        else:
            for hook in ctx.original_hook_on_batch_start_eval:
                hook(ctx)
        ctx.data_batch_cache = copy.deepcopy(ctx.data_batch)
    else:
        # reuse the data_cache since the original hook `_hook_on_batch_end`
        # will clean `data_batch`
        ctx.data_batch = copy.deepcopy(ctx.data_batch_cache)
    ctx.pFedMe_approx_fit_counter = (ctx.pFedMe_approx_fit_counter +
                                     1) % ctx.pFedMe_K


def _hook_on_batch_end_flop_count(ctx):
    # besides the normal forward flops, pFedMe introduces
    # 1) the regularization adds the cost of number of model parameters
    ctx.monitor.total_flops += ctx.monitor.total_model_size / 2


def _hook_on_epoch_end_flop_count(ctx):
    # due to the local weight updating
    ctx.monitor.total_flops += ctx.monitor.total_model_size / 2


def hook_on_epoch_end_update_local(ctx):
    # update local weight after finding approximate theta
    for client_param, local_para_tmp in zip(
            ctx.model.parameters(), ctx.pFedMe_local_model_tmp.parameters()):
        local_para_tmp.data = local_para_tmp.data - \
                              ctx.optimizer.regular_weight * \
                              ctx.pFedMe_outer_lr * (local_para_tmp.data -
                                                     client_param.data)

    # set the compared model data, then the optimizer will find approximate
    # model using trainer.cfg.personalization.lr
    compared_global_model_para = [{
        "params": list(ctx.pFedMe_local_model_tmp.parameters())
    }]
    ctx.optimizer.set_compared_para_group(compared_global_model_para)


def hook_on_fit_end_update_local(ctx):
    for param, local_para_tmp in zip(ctx.model.parameters(),
                                     ctx.pFedMe_local_model_tmp.parameters()):
        param.data = local_para_tmp.data

    del ctx.pFedMe_local_model_tmp
