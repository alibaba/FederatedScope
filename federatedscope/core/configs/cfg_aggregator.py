from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_aggregator_cfg(cfg):
    cfg.aggregator = CN()
    cfg.aggregator.num_agg_groups = 1
    cfg.aggregator.num_agg_topk = []
    cfg.aggregator.inside_weight = 1.0
    cfg.aggregator.outside_weight = 0.0

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_aggregator_cfg)


def assert_aggregator_cfg(cfg):
    pass


register_config('aggregator', extend_aggregator_cfg)
