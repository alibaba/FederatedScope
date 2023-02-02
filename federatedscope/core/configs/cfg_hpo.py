import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config
from federatedscope.core.monitors.metric_calculator import SUPPORT_METRICS

logger = logging.getLogger(__name__)


def extend_hpo_cfg(cfg):

    # ---------------------------------------------------------------------- #
    # hpo related options
    # ---------------------------------------------------------------------- #
    cfg.hpo = CN()
    cfg.hpo.working_folder = 'hpo'
    cfg.hpo.ss = ''
    cfg.hpo.num_workers = 0
    cfg.hpo.init_cand_num = 16
    cfg.hpo.larger_better = False
    cfg.hpo.scheduler = 'rs'
    cfg.hpo.metric = 'client_summarized_weighted_avg.val_loss'

    # SHA
    cfg.hpo.sha = CN()
    cfg.hpo.sha.elim_rate = 3
    cfg.hpo.sha.budgets = []
    cfg.hpo.sha.iter = 0

    # PBT
    cfg.hpo.pbt = CN()
    cfg.hpo.pbt.max_stage = 5
    cfg.hpo.pbt.perf_threshold = 0.1

    # FedEx
    cfg.hpo.fedex = CN()
    cfg.hpo.fedex.use = False
    cfg.hpo.fedex.ss = ''
    cfg.hpo.fedex.flatten_ss = True
    # If <= .0, use 'auto'
    cfg.hpo.fedex.eta0 = -1.0
    cfg.hpo.fedex.sched = 'auto'
    # cutoff: entropy level below which to stop updating the config
    # probability and use MLE
    cfg.hpo.fedex.cutoff = .0
    # discount factor; 0.0 is most recent, 1.0 is mean
    cfg.hpo.fedex.gamma = .0
    cfg.hpo.fedex.diff = False
    cfg.hpo.fedex.psn = False
    cfg.hpo.fedex.pi_lr = 0.01

    # Table
    cfg.hpo.table = CN()
    cfg.hpo.table.eps = 0.1
    cfg.hpo.table.num = 27
    cfg.hpo.table.idx = 0

    # HPO Diagnosis
    cfg.hpo.diagnosis = CN()
    cfg.hpo.diagnosis.use = False
    cfg.hpo.diagnosis.landscape_1d = []  # From one hyper_param, eg. [lr, wd]

    cfg.hpo.trial_index = 0

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_hpo_cfg)


def assert_hpo_cfg(cfg):
    for key, value in SUPPORT_METRICS.items():
        is_larger_the_better = value[1]
        if key in cfg.hpo.metric and is_larger_the_better != \
                cfg.hpo.larger_better:
            logger.warning(f'`cfg.hpo.larger_better` is overwritten by '
                           f'{is_larger_the_better} for the metric `'
                           f'{cfg.hpo.metric}` is  {is_larger_the_better} '
                           f'for larger the better.')
            cfg.hpo.larger_better = is_larger_the_better
            break

    assert cfg.hpo.num_workers >= 0, "#worker should be non-negative but " \
                                     "given {}".format(cfg.hpo.num_workers)
    assert len(cfg.hpo.sha.budgets) > 0, \
        "Either do NOT specify the budgets or specify the budget for each " \
        "SHA iteration, but the given budgets is {}".format(
            cfg.hpo.sha.budgets)

    assert not (cfg.hpo.fedex.use and cfg.federate.use_ss
                ), "Cannot use secret sharing and FedEx at the same time"
    assert cfg.train.optimizer.type == 'SGD' or not cfg.hpo.fedex.use, \
        "SGD is required if FedEx is considered"
    assert cfg.hpo.fedex.sched in [
        'adaptive', 'aggressive', 'auto', 'constant', 'scale'
    ], "schedule of FedEx must be choice from {}".format(
        ['adaptive', 'aggressive', 'auto', 'constant', 'scale'])
    assert cfg.hpo.fedex.gamma >= .0 and cfg.hpo.fedex.gamma <= 1.0, \
        "{} must be in [0, 1]".format(cfg.hpo.fedex.gamma)
    assert cfg.hpo.fedex.use == cfg.federate.use_diff, "Once FedEx is " \
                                                       "adopted, " \
                                                       "federate.use_diff " \
                                                       "must be True."


register_config("hpo", extend_hpo_cfg)
