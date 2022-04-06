from federatedscope.core.trainers.trainer import GeneralTorchTrainer
from typing import Type

from federatedscope.attack.privacy_attacks.GAN_based_attack import GANCRA


def wrap_GANTrainer(
        base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:
    '''
    Warp the trainer for gan_based class representative attack.

    Args:
        base_trainer: Type: core.trainers.GeneralTrainer

    Returns:
        The wrapped trainer; Type: core.trainers.GeneralTrainer

    '''

    # ---------------- attribute-level plug-in -----------------------

    base_trainer.ctx.target_label_ind = base_trainer.cfg.attack.target_label_ind
    base_trainer.ctx.gan_cra = GANCRA(base_trainer.cfg.attack.target_label_ind,
                                      base_trainer.ctx.model,
                                      dataset_name=base_trainer.cfg.data.type,
                                      device=base_trainer.ctx.device,
                                      sav_pth=base_trainer.cfg.outdir)

    # ---- action-level plug-in -------

    base_trainer.register_hook_in_train(new_hook=hood_on_fit_start_generator,
                                        trigger='on_fit_start',
                                        insert_mode=-1)
    base_trainer.register_hook_in_train(new_hook=hook_on_gan_cra_train,
                                        trigger='on_batch_start',
                                        insert_mode=-1)
    base_trainer.register_hook_in_train(
        new_hook=hook_on_batch_injected_data_generation,
        trigger='on_batch_start',
        insert_mode=-1)
    base_trainer.register_hook_in_train(
        new_hook=hook_on_batch_forward_injected_data,
        trigger='on_batch_forward',
        insert_mode=-1)

    base_trainer.register_hook_in_train(
        new_hook=hook_on_data_injection_sav_data,
        trigger='on_fit_end',
        insert_mode=-1)

    return base_trainer


def hood_on_fit_start_generator(ctx):
    ctx.gan_cra.round_num += 1
    print('----- Round {}: GAN training ............'.format(
        ctx.gan_cra.round_num))


def hook_on_batch_forward_injected_data(ctx):
    # inject the generated data into training batch loss
    x, label = [_.to(ctx.device) for _ in ctx.data_batch]
    pred = ctx.model(x)
    if len(label.size()) == 0:
        label = label.unsqueeze(0)
    ctx.loss_batch += ctx.criterion(pred, label)
    ctx.y_true_injected = label
    ctx.y_prob_injected = pred


def hook_on_batch_injected_data_generation(ctx):
    # generate the injected data
    ctx.injected_data = ctx.gan_cra.generate_fake_data()


def hook_on_gan_cra_train(ctx):

    ctx.gan_cra.update_discriminator(ctx.model)
    ctx.gan_cra.generator_train()


def hook_on_data_injection_sav_data(ctx):

    ctx.gan_cra.generate_and_save_images()
