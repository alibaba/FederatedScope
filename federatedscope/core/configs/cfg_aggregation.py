import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_aggregation_cfg(cfg):

    # ---------------------------------------------------------------------- #
    # aggregation related options
    # ---------------------------------------------------------------------- #
    cfg.aggregation = CN()
    cfg.aggregation.byzantine_node_num = 0

    # Krum/multi-Krum
    cfg.aggregation.krum = CN()
    cfg.aggregation.krum.use = False
    cfg.aggregation.krum.agg_num = 1

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_aggregation_cfg)


def assert_aggregation_cfg(cfg):

    if cfg.aggregation.byzantine_node_num == 0 and cfg.aggregation.krum.use:
        logging.warning('Although krum aggregtion rule is applied, we found '
                        'that cfg.aggregation.byzantine_node_num == 0')


register_config("aggregation", extend_aggregation_cfg)
