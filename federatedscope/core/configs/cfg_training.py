from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_training_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # Trainer related options
    # ---------------------------------------------------------------------- #
    cfg.trainer = CN()

    cfg.trainer.type = 'general'

    # ---------------------------------------------------------------------- #
    # Training related options
    # ---------------------------------------------------------------------- #
    cfg.train = CN()

    cfg.train.local_update_steps = 1
    cfg.train.batch_or_epoch = 'batch'

    cfg.train.optimizer = CN(new_allowed=True)
    cfg.train.optimizer.type = 'SGD'
    cfg.train.optimizer.lr = 0.1

    # ---------------------------------------------------------------------- #
    # Finetune related options
    # ---------------------------------------------------------------------- #
    cfg.finetune = CN()

    cfg.finetune.before_eval = False
    cfg.finetune.local_update_steps = 1
    cfg.finetune.batch_or_epoch = 'epoch'
    cfg.finetune.freeze_param = ""

    cfg.finetune.optimizer = CN(new_allowed=True)
    cfg.finetune.optimizer.type = 'SGD'
    cfg.finetune.optimizer.lr = 0.1

    # ---------------------------------------------------------------------- #
    # Gradient related options
    # ---------------------------------------------------------------------- #
    cfg.grad = CN()
    cfg.grad.grad_clip = -1.0  # negative numbers indicate we do not clip grad

    # ---------------------------------------------------------------------- #
    # Early stopping related options
    # ---------------------------------------------------------------------- #
    cfg.early_stop = CN()

    # patience (int): How long to wait after last time the monitored metric
    # improved.
    # Note that the actual_checking_round = patience * cfg.eval.freq
    # To disable the early stop, set the early_stop.patience a integer <=0
    cfg.early_stop.patience = 5
    # delta (float): Minimum change in the monitored metric to indicate an
    # improvement.
    cfg.early_stop.delta = 0.0
    # Early stop when no improve to last `patience` round, in ['mean', 'best']
    cfg.early_stop.improve_indicator_mode = 'best'
    cfg.early_stop.the_smaller_the_better = True

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_training_cfg)


def assert_training_cfg(cfg):
    if cfg.train.batch_or_epoch not in ['batch', 'epoch']:
        raise ValueError(
            "Value of 'cfg.train.batch_or_epoch' must be chosen from ["
            "'batch', 'epoch'].")

    if cfg.finetune.batch_or_epoch not in ['batch', 'epoch']:
        raise ValueError(
            "Value of 'cfg.finetune.batch_or_epoch' must be chosen from ["
            "'batch', 'epoch'].")

    # TODO: should not be here?
    if cfg.backend not in ['torch', 'tensorflow']:
        raise ValueError(
            "Value of 'cfg.backend' must be chosen from ['torch', "
            "'tensorflow'].")
    if cfg.backend == 'tensorflow' and cfg.federate.mode == 'standalone':
        raise ValueError(
            "We only support run with distribued mode when backend is "
            "tensorflow")
    if cfg.backend == 'tensorflow' and cfg.use_gpu is True:
        raise ValueError(
            "We only support run with cpu when backend is tensorflow")

    if cfg.finetune.before_eval is False and cfg.finetune.local_update_steps\
            <= 0:
        raise ValueError(
            f"When adopting fine-tuning, please set a valid local fine-tune "
            f"steps, got {cfg.finetune.local_update_steps}")


register_config("fl_training", extend_training_cfg)
