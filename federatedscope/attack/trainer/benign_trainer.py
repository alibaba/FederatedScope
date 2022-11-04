import logging
from typing import Type
import numpy as np

from federatedscope.core.trainers import GeneralTorchTrainer

logger = logging.getLogger(__name__)


def wrap_benignTrainer(
        base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:
    '''
    Warp the benign trainer for backdoor attack:
    We just add the normalization operation.
    Args:
        base_trainer: Type: core.trainers.GeneralTorchTrainer
    :returns:
        The wrapped trainer; Type: core.trainers.GeneralTorchTrainer
    '''
    base_trainer.register_hook_in_eval(new_hook=hook_on_fit_end_test_poison,
                                       trigger='on_fit_end',
                                       insert_pos=0)

    return base_trainer


def hook_on_fit_end_test_poison(ctx):
    """
    Evaluate metrics of poisoning attacks.
    """

    ctx['poison_' + ctx.cur_split + '_loader'] = ctx.data['poison_' +
                                                          ctx.cur_split]
    ctx['poison_' + ctx.cur_split + '_data'] = ctx.data['poison_' +
                                                        ctx.cur_split].dataset
    ctx['num_poison_' + ctx.cur_split + '_data'] = len(
        ctx.data['poison_' + ctx.cur_split].dataset)
    setattr(ctx, "poison_{}_y_true".format(ctx.cur_split), [])
    setattr(ctx, "poison_{}_y_prob".format(ctx.cur_split), [])
    setattr(ctx, "poison_num_samples_{}".format(ctx.cur_split), 0)

    for batch_idx, (samples, targets) in enumerate(
            ctx['poison_' + ctx.cur_split + '_loader']):
        samples, targets = samples.to(ctx.device), targets.to(ctx.device)
        pred = ctx.model(samples)
        if len(targets.size()) == 0:
            targets = targets.unsqueeze(0)
        ctx.poison_y_true = targets
        ctx.poison_y_prob = pred
        ctx.poison_batch_size = len(targets)

        ctx.get("poison_{}_y_true".format(ctx.cur_split)).append(
            ctx.poison_y_true.detach().cpu().numpy())

        ctx.get("poison_{}_y_prob".format(ctx.cur_split)).append(
            ctx.poison_y_prob.detach().cpu().numpy())

        setattr(
            ctx, "poison_num_samples_{}".format(ctx.cur_split),
            ctx.get("poison_num_samples_{}".format(ctx.cur_split)) +
            ctx.poison_batch_size)

    setattr(ctx, "poison_{}_y_true".format(ctx.cur_split),
            np.concatenate(ctx.get("poison_{}_y_true".format(ctx.cur_split))))
    setattr(ctx, "poison_{}_y_prob".format(ctx.cur_split),
            np.concatenate(ctx.get("poison_{}_y_prob".format(ctx.cur_split))))

    logger.info('the {} poisoning samples: {:d}'.format(
        ctx.cur_split, ctx.get("poison_num_samples_{}".format(ctx.cur_split))))

    poison_true = ctx['poison_' + ctx.cur_split + '_y_true']
    poison_prob = ctx['poison_' + ctx.cur_split + '_y_prob']

    poison_pred = np.argmax(poison_prob, axis=1)

    correct = poison_true == poison_pred

    poisoning_acc = float(np.sum(correct)) / len(correct)

    logger.info('the {} poisoning accuracy: {:f}'.format(
        ctx.cur_split, poisoning_acc))
