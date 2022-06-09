from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_maml_cfg(cfg):
    # ------------------------------------------------------------------------ #
    # Dataset related options
    # ------------------------------------------------------------------------ #
    cfg.maml = CN()

    cfg.maml.use = False
    cfg.maml.inner_lr = 0.01

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_maml_cfg)


def assert_maml_cfg(cfg):
    pass

register_config("maml", extend_maml_cfg)