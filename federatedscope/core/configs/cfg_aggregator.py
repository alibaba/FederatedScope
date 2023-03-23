import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_aggregator_cfg(cfg):

    # ---------------------------------------------------------------------- #
    # aggregator related options
    # fs has supported the robust aggregation rules of 'krum', 'median',
    # 'trimmedmean', 'bulyan' and 'normbounding', the use case is refered
    #  to tests/test_robust_aggregators.py
    # ---------------------------------------------------------------------- #
    cfg.aggregator = CN(new_allowed=True)
    cfg.aggregator.robust_rule = 'oracle_fedavg'
    cfg.aggregator.byzantine_node_num = 0

    # For krum Algos
    cfg.aggregator.krum = CN()
    cfg.aggregator.krum.agg_num = 1

    # For trimmedmean Algos
    cfg.aggregator.trimmedmean = CN()
    cfg.aggregator.trimmedmean.excluded_ratio = 0.1

    # For normbounding Algos
    cfg.aggregator.normbounding = CN()
    cfg.aggregator.normbounding.norm_bound = 10.0

    # For ATC method
    cfg.aggregator.num_agg_groups = 1
    cfg.aggregator.num_agg_topk = []
    cfg.aggregator.inside_weight = 1.0
    cfg.aggregator.outside_weight = 0.0

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_aggregator_cfg)


def assert_aggregator_cfg(cfg):

    if cfg.aggregator.byzantine_node_num == 0 and \
            cfg.aggregator.robust_rule in \
            ['krum', 'normbounding', 'median', 'trimmedmean', 'bulyan']:
        logging.warning(
            f'Although {cfg.aggregator.robust_rule} aggregtion rule is '
            'applied, we found that cfg.aggregator.byzantine_node_num == 0')


register_config('aggregator', extend_aggregator_cfg)
