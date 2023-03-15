import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_aggregator_cfg(cfg):

    # ---------------------------------------------------------------------- #
    # aggregator related options
    # ---------------------------------------------------------------------- #
    cfg.aggregator = CN()
    cfg.aggregator.byzantine_node_num = 0
    cfg.aggregator.client_sampled_ratio = 0.2

    # For fedavg Algos
    cfg.aggregator.fedavg = CN()
    cfg.aggregator.fedavg.use = False

    # For krum/multi-krum Algos
    cfg.aggregator.krum = CN()
    cfg.aggregator.krum.use = False
    cfg.aggregator.krum.agg_num = 1

     # For median Algos
    cfg.aggregator.median = CN()
    cfg.aggregator.median.use = False

     # For trimmed_mean Algos
    cfg.aggregator.trimmedmean = CN()
    cfg.aggregator.trimmedmean.use = False
    cfg.aggregator.trimmedmean.excluded_ratio=0.1

     # For bulyan Algos
    cfg.aggregator.bulyan = CN()
    cfg.aggregator.bulyan.use = False

     # For fltrust Algos
    cfg.aggregator.fltrust = CN()
    cfg.aggregator.fltrust.use = False
    cfg.aggregator.fltrust.global_learningrate = 0.01

    # For normbounding Algos
    cfg.aggregator.normbounding = CN()
    cfg.aggregator.normbounding.use = False
    cfg.aggregator.normbounding.norm_bound = 10.0

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

