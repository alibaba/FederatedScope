def wrap_attacker_trainer(base_trainer, config):
    '''
    Wrap the trainer for attack client.
    Args:
        base_trainer: the trainer that will be wrapped; Type: core.trainers.GeneralTrainer
        config: the configure; Type: yacs.config.CfgNode

    Returns:
        The wrapped trainer; Type: core.trainers.GeneralTrainer

    '''
    if config.attack.attack_method.lower() == 'gan_attack':
        from federatedscope.attack.trainer.GAN_trainer import wrap_GANTrainer
        return wrap_GANTrainer(base_trainer)
    elif config.attack.attack_method.lower() == 'gradascent':
        from federatedscope.attack.trainer.MIA_invert_gradient_trainer import wrap_GradientAscentTrainer
        return wrap_GradientAscentTrainer(base_trainer)

    else:
        raise ValueError('Trainer {} is not provided'.format(
            config.attack.attack_method))