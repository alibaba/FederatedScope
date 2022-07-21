from calendar import c
import logging
from typing import Type
import torch
import numpy as np

from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.auxiliaries.transform_builder import get_transform
from federatedscope.attack.auxiliary.backdoor_utils import normalize
from federatedscope.core.auxiliaries.dataloader_builder import WrapDataset
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.auxiliaries.ReIterator import ReIterator

# from federatedscope.attack.auxiliary.poisoning_dataset import load_poisoned_dataset_edgeset, load_poisoned_dataset_pixel

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

    # ---- action-level plug-in -------
    
    # base_trainer.register_hook_in_train(new_hook=hook_on_fit_start_addnormalize,
    #                                     trigger='on_fit_start',
    #                                     insert_pos=0)
    

    # base_trainer.register_hook_in_eval(new_hook=hook_on_fit_start_addnormalize,
    #                                     trigger='on_fit_start',
    #                                     insert_pos=0)

    # base_trainer.register_hook_in_eval(new_hook=hook_on_fit_start_test_poison,
    #                                     trigger='on_fit_start',
    #                                     insert_pos=-1)

    # base_trainer.register_hook_in_eval(new_hook=hook_on_epoch_start_test_poison,
    #                                     trigger='on_epoch_start',
    #                                     insert_pos=-1)


    # base_trainer.register_hook_in_eval(new_hook=hook_on_batch_start_test_poison,
    #                                     trigger='on_batch_start',
    #                                     insert_pos=-1)


    # base_trainer.register_hook_in_eval(new_hook=hook_on_batch_forward_test_poison,
    #                                     trigger='on_batch_forward',
    #                                     insert_pos=-1)

    # base_trainer.register_hook_in_eval(new_hook=hook_on_batch_end_test_poison, 
    #                                     trigger="on_batch_end",
    #                                     insert_pos=-1)


    base_trainer.register_hook_in_eval(new_hook=hook_on_fit_end_test_poison,
                                        trigger='on_fit_end',
                                        insert_pos=0)
    

    

    return base_trainer




# def hook_on_fit_start_test_poison(ctx):


#     ctx['poison_'+ctx.cur_data_split+'_loader'] = ctx.data['poison_'+ctx.cur_data_split]
#     ctx['poison_'+ctx.cur_data_split+'_data'] = ctx.data['poison_'+ctx.cur_data_split].dataset
#     ctx['num_poison_'+ctx.cur_data_split+'_data'] = len(ctx.data['poison_'+ctx.cur_data_split].dataset)
#     setattr(ctx, "poison_loss_batch_total_{}".format(ctx.cur_data_split), 0)
#     setattr(ctx, "poison_num_samples_{}".format(ctx.cur_data_split), 0)
#     setattr(ctx, "poison_{}_y_true".format(ctx.cur_data_split), [])
#     setattr(ctx, "poison_{}_y_prob".format(ctx.cur_data_split), [])




# def hook_on_epoch_start_test_poison(ctx):
#     # prepare dataloader
#     if ctx.get("poison_{}_loader".format(ctx.cur_data_split)) is None:
#         loader = get_dataloader(
#             WrapDataset(ctx.get("poison_{}_data".format(ctx.cur_data_split))),
#             ctx.cfg)
#         setattr(ctx, "poison_{}_loader".format(ctx.cur_data_split),
#                 ReIterator(loader))
#     elif not isinstance(ctx.get("poison_{}_loader".format(ctx.cur_data_split)),
#                         ReIterator):
#         setattr(
#             ctx, "poison_{}_loader".format(ctx.cur_data_split),
#             ReIterator(ctx.get("poison_{}_loader".format(ctx.cur_data_split))))
#     else:
#         ctx.get("poison_{}_loader".format(ctx.cur_data_split)).reset()
        


# def hook_on_batch_start_test_poison(ctx):
#     # prepare data batch
#     try:
#         ctx.poison_data_batch = next(
#             ctx.get("poison_{}_loader".format(ctx.cur_data_split)))
#     except StopIteration:
#         raise StopIteration


# def hook_on_batch_forward_test_poison(ctx):

#     x, label = [_.to(ctx.device) for _ in ctx.poison_data_batch]
#     pred = ctx.model(x)
#     if len(label.size()) == 0:
#         label = label.unsqueeze(0)
#     ctx.poison_loss_batch = ctx.criterion(pred, label)
#     ctx.poison_y_true = label
#     ctx.poison_y_prob = pred

#     ctx.poison_batch_size = len(label)



# def hook_on_batch_end_test_poison(ctx):

#     # update statistics
#     setattr(
#         ctx, "poison_loss_batch_total_{}".format(ctx.cur_data_split),
#         ctx.get("poison_loss_batch_total_{}".format(ctx.cur_data_split)) +
#         ctx.poison_loss_batch.item() * ctx.poison_batch_size)


#     setattr(
#         ctx, "poison_num_samples_{}".format(ctx.cur_data_split),
#         ctx.get("poison_num_samples_{}".format(ctx.cur_data_split)) +
#         ctx.poison_batch_size)

#     # cache label for evaluate
#     ctx.get("poison_{}_y_true".format(ctx.cur_data_split)).append(
#         ctx.poison_y_true.detach().cpu().numpy())

#     ctx.get("poison_{}_y_prob".format(ctx.cur_data_split)).append(
#         ctx.poison_y_prob.detach().cpu().numpy())

#     # clean temp ctx
#     ctx.poison_data_batch = None
#     ctx.poison_batch_size = None
#     ctx.poison_loss_task = None
#     ctx.poison_loss_batch = None
#     ctx.poison_loss_regular = None
#     ctx.poison_y_true = None
#     ctx.poison_y_prob = None



def hook_on_fit_end_test_poison(ctx):

    """Evaluate metrics of poisoning attacks.

    """

    ctx['poison_'+ctx.cur_data_split+'_loader'] = ctx.data['poison_'+ctx.cur_data_split]
    ctx['poison_'+ctx.cur_data_split+'_data'] = ctx.data['poison_'+ctx.cur_data_split].dataset
    ctx['num_poison_'+ctx.cur_data_split+'_data'] = len(ctx.data['poison_'+ctx.cur_data_split].dataset)
    setattr(ctx, "poison_{}_y_true".format(ctx.cur_data_split), [])
    setattr(ctx, "poison_{}_y_prob".format(ctx.cur_data_split), [])
    setattr(ctx, "poison_num_samples_{}".format(ctx.cur_data_split), 0)

    for batch_idx, (samples, targets) in enumerate(ctx['poison_'+ctx.cur_data_split+'_loader']):
        samples, targets = samples.to(ctx.device), targets.to(ctx.device)
        pred = ctx.model(samples)
        if len(targets.size()) == 0:
            targets = targets.unsqueeze(0)
        ctx.poison_y_true = targets
        ctx.poison_y_prob = pred
        ctx.poison_batch_size = len(targets)

        ctx.get("poison_{}_y_true".format(ctx.cur_data_split)).append(
            ctx.poison_y_true.detach().cpu().numpy())

        ctx.get("poison_{}_y_prob".format(ctx.cur_data_split)).append(
            ctx.poison_y_prob.detach().cpu().numpy())
    
        setattr(
            ctx, "poison_num_samples_{}".format(ctx.cur_data_split),
            ctx.get("poison_num_samples_{}".format(ctx.cur_data_split)) +
            ctx.poison_batch_size)
            


    # import pdb; pdb.set_trace()

    setattr(
        ctx, "poison_{}_y_true".format(ctx.cur_data_split),
        np.concatenate(ctx.get("poison_{}_y_true".format(ctx.cur_data_split))))
    setattr(
        ctx, "poison_{}_y_prob".format(ctx.cur_data_split),
        np.concatenate(ctx.get("poison_{}_y_prob".format(ctx.cur_data_split))))

    logger.info('the {} poisoning samples: {:d}'.format(ctx.cur_data_split, ctx.get("poison_num_samples_{}".format(ctx.cur_data_split))))

    poison_true = ctx['poison_'+ctx.cur_data_split+'_y_true']
    poison_prob = ctx['poison_'+ctx.cur_data_split+'_y_prob']

    poison_pred = np.argmax(poison_prob, axis=1)

    correct = poison_true == poison_pred

    poisoning_acc = float(np.sum(correct)) / len(correct)

    logger.info('the {} poisoning accuracy: {:f}'.format(ctx.cur_data_split, poisoning_acc))
