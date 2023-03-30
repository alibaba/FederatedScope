import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_aggregator_cfg(cfg):

    # ---------------------------------------------------------------------- #
    # aggregator related options
    # ---------------------------------------------------------------------- #
    cfg.aggregator = CN()
    cfg.aggregator.byzantine_node_num = 0

    # For krum/multi-krum Algos
    cfg.aggregator.krum = CN()
    cfg.aggregator.krum.use = False
    cfg.aggregator.krum.agg_num = 1

    # For ATC method
    cfg.aggregator.num_agg_groups = 1
    cfg.aggregator.num_agg_topk = []
    cfg.aggregator.inside_weight = 1.0
    cfg.aggregator.outside_weight = 0.0

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_aggregator_cfg)


def assert_aggregator_cfg(cfg):

    if cfg.aggregator.byzantine_node_num == 0 and cfg.aggregator.krum.use:
        logging.warning('Although krum aggregtion rule is applied, we found '
                        'that cfg.aggregator.byzantine_node_num == 0')


register_config('aggregator', extend_aggregator_cfg)
