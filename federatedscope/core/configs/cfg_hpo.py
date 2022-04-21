from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_hpo_cfg(cfg):

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

    cfg.hpo.fedex = CN()
    cfg.hpo.fedex.use = False
    cfg.hpo.fedex.ss = ''
    cfg.hpo.fedex.num_arms = 16


def assert_hpo_cfg(cfg):
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

    assert not (cfg.hpo.fedex.use and cfg.federate.use_ss), "Cannot use secret sharing and FedEx at the same time"
    assert cfg.optimizer.type == 'SGD' or not cfg.hpo.fedex.use, "SGD is required if FedEx is considered"


register_config("hpo", extend_hpo_cfg)
