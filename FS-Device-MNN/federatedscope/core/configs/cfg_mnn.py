import os

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_mnn_cfg(cfg):
    cfg.mnn = CN()
    cfg.mnn.cmd_convert = ''


def assert_mnn_cfg(cfg):
    assert os.path.exists(cfg.mnn.cmd_convert), f"Cannot find {cfg.mnn.cmd_convert}!"


register_config("mnn", extend_mnn_cfg)
