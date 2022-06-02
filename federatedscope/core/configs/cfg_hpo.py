from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_hpo_cfg(cfg):

    # ------------------------------------------------------------------------ #
    # hpo related options
    # ------------------------------------------------------------------------ #
    cfg.hpo = CN()
    cfg.hpo.working_folder = 'hpo'
    cfg.hpo.ss = ''
    cfg.hpo.num_workers = 1
    #cfg.hpo.init_strategy = 'random'
    cfg.hpo.init_cand_num = 16
    cfg.hpo.log_scale = False
    cfg.hpo.larger_better = False
    cfg.hpo.scheduler = 'rs'
    # plot the performance
    cfg.hpo.plot_interval = 1
    cfg.hpo.metric = 'client_summarized_weighted_avg.val_loss'

    # SHA
    cfg.hpo.sha = CN()
    cfg.hpo.sha.elim_round_num = 3
    cfg.hpo.sha.elim_rate = 3
    cfg.hpo.sha.budgets = []

    # PBT
    cfg.hpo.pbt = CN()
    cfg.hpo.pbt.max_stage = 5
    cfg.hpo.pbt.perf_threshold = 0.1

    # FedEx
    cfg.hpo.fedex = CN()
    cfg.hpo.fedex.use = False
    cfg.hpo.fedex.ss = ''
    cfg.hpo.fedex.flatten_ss = True
    cfg.hpo.fedex.num_arms = 16
    cfg.hpo.fedex.diff = False
    cfg.hpo.fedex.sched = 'auto'
    # discount factor; 0.0 is most recent, 1.0 is mean
    cfg.hpo.fedex.gamma = .0
    # cutoff: entropy level below which to stop updating the config probability and use MLE
    cfg.hpo.fedex.cutoff = .0


def assert_hpo_cfg(cfg):
    # HPO related
    #assert cfg.hpo.init_strategy in [
    #    'full', 'grid', 'random'
    #], "initialization strategy for HPO should be \"full\", \"grid\", or \"random\", but the given choice is {}".format(
    #    cfg.hpo.init_strategy)
    assert cfg.hpo.scheduler in ['rs', 'sha',
                                 'pbt'], "No HPO scheduler named {}".format(
                                     cfg.hpo.scheduler)
    assert len(cfg.hpo.sha.budgets) == 0 or len(
        cfg.hpo.sha.budgets
    ) == cfg.hpo.sha.elim_round_num, \
        "Either do NOT specify the budgets or specify the budget for each SHA iteration, but the given budgets is {}".\
            format(cfg.hpo.sha.budgets)

    assert not (cfg.hpo.fedex.use and cfg.federate.use_ss
                ), "Cannot use secret sharing and FedEx at the same time"
    assert cfg.optimizer.type == 'SGD' or not cfg.hpo.fedex.use, "SGD is required if FedEx is considered"
    assert cfg.hpo.fedex.sched in [
        'adaptive', 'aggressive', 'auto', 'constant', 'scale'
    ], "schedule of FedEx must be choice from {}".format(
        ['adaptive', 'aggressive', 'auto', 'constant', 'scale'])
    assert cfg.hpo.fedex.gamma >= .0 and cfg.hpo.fedex.gamma <= 1.0, "{} must be in [0, 1]".format(
        cfg.hpo.fedex.gamma)
    assert cfg.hpo.fedex.diff == cfg.federate.use_diff, "Inconsistent values for cfg.hpo.fedex.diff={} and cfg.federate.use_diff={}".format(
        cfg.hpo.fedex.diff, cfg.federate.use_diff)


register_config("hpo", extend_hpo_cfg)
