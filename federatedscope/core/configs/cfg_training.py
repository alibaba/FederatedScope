import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_training_cfg(cfg):
    # ------------------------------------------------------------------------ #
    # Trainer related options
    # ------------------------------------------------------------------------ #
    cfg.trainer = CN()

    cfg.trainer.type = 'general'
    cfg.trainer.finetune = CN()
    cfg.trainer.finetune.steps = 0
    cfg.trainer.finetune.only_psn = True
    cfg.trainer.finetune.stepsize = 0.01

    # ------------------------------------------------------------------------ #
    # Optimizer related options
    # ------------------------------------------------------------------------ #
    cfg.optimizer = CN()

    cfg.optimizer.type = 'SGD'
    cfg.optimizer.lr = 0.1
    cfg.optimizer.weight_decay = .0
    cfg.optimizer.grad_clip = -1.0  # negative numbers indicate we do not clip grad

    # ------------------------------------------------------------------------ #
    # lr_scheduler related options
    # ------------------------------------------------------------------------ #
    # cfg.lr_scheduler = CN()
    # cfg.lr_scheduler.type = 'StepLR'
    # cfg.lr_scheduler.schlr_params = dict()

    # ------------------------------------------------------------------------ #
    # Early stopping related options
    # ------------------------------------------------------------------------ #
    cfg.early_stop = CN()

    # patience (int): How long to wait after last time the monitored metric improved.
    # Note that the actual_checking_round = patience * cfg.eval.freq
    # To disable the early stop, set the early_stop.patience a integer <=0
    cfg.early_stop.patience = 5
    # delta (float): Minimum change in the monitored metric to indicate an improvement.
    cfg.early_stop.delta = 0.0
    # Early stop when no improve to last `patience` round, in ['mean', 'best']
    cfg.early_stop.improve_indicator_mode = 'best'
    cfg.early_stop.the_smaller_the_better = True

    # ------------------------------------------------------------------------ #
    # hpo related options
    # ------------------------------------------------------------------------ #
    cfg.hpo = CN()
    cfg.hpo.working_folder = 'hpo'
    cfg.hpo.init_strategy = 'random'
    cfg.hpo.init_cand_num = 16
    cfg.hpo.log_scale = False
    cfg.hpo.larger_better = False
    cfg.hpo.scheduler = 'bruteforce'
    # plot the performance
    cfg.hpo.plot_interval = 1
    cfg.hpo.metric = 'client_summarized_weighted_avg.test_loss'
    cfg.hpo.sha = CN()
    cfg.hpo.sha.elim_round_num = 3
    cfg.hpo.sha.elim_rate = 3
    cfg.hpo.sha.budgets = []
    cfg.hpo.pbt = CN()
    cfg.hpo.pbt.max_stage = 5
    cfg.hpo.pbt.perf_threshold = 0.1

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_training_cfg)


def assert_training_cfg(cfg):
    # HPO related
    assert cfg.hpo.init_strategy in [
        'full', 'grid', 'random'
    ], "initialization strategy for HPO should be \"full\", \"grid\", or \"random\", but the given choice is {}".format(
        cfg.hpo.init_strategy)
    assert cfg.hpo.scheduler in ['bruteforce', 'sha',
                                 'pbt'], "No HPO scheduler named {}".format(
                                     cfg.hpo.scheduler)
    assert len(cfg.hpo.sha.budgets) == 0 or len(
        cfg.hpo.sha.budgets
    ) == cfg.hpo.sha.elim_round_num, \
        "Either do NOT specify the budgets or specify the budget for each SHA iteration, but the given budgets is {}".\
            format(cfg.hpo.sha.budgets)

    if cfg.backend not in ['torch', 'tensorflow']:
        raise ValueError(
            "Value of 'cfg.backend' must be chosen from ['torch', 'tensorflow']."
        )
    if cfg.backend == 'tensorflow' and cfg.federate.mode == 'standalone':
        raise ValueError(
            "We only support run with distribued mode when backend is tensorflow"
        )
    if cfg.backend == 'tensorflow' and cfg.use_gpu == True:
        raise ValueError(
            "We only support run with cpu when backend is tensorflow")


register_config("fl_training", extend_training_cfg)
