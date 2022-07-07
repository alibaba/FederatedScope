def wrap_attacker_trainer(base_trainer, config):
    '''Wrap the trainer for attack client.
    Args:
        base_trainer (core.trainers.GeneralTorchTrainer): the trainer that
        will be wrapped;
        config (yacs.config.CfgNode): the configure;

    :returns:
        The wrapped trainer; Type: core.trainers.GeneralTorchTrainer

    '''
    if config.attack.attack_method.lower() == 'gan_attack':
        from federatedscope.attack.trainer import wrap_GANTrainer
        return wrap_GANTrainer(base_trainer)
    elif config.attack.attack_method.lower() == 'gradascent':
        from federatedscope.attack.trainer import wrap_GradientAscentTrainer
        return wrap_GradientAscentTrainer(base_trainer)
    else:
        raise ValueError('Trainer {} is not provided'.format(
            config.attack.attack_method))
