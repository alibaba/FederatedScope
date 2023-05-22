import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config
import torch

logger = logging.getLogger(__name__)


def extend_llm_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # Adapters for LLM
    # ---------------------------------------------------------------------- #
    cfg.adapter = CN()
    cfg.adapter.use = False
    cfg.adapter.args = []


def assert_llm_cfg(cfg):
    pass


register_config("llm", extend_llm_cfg)
