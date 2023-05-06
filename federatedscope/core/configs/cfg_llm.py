import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

logger = logging.getLogger(__name__)


def extend_llm_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # LLM related options
    # ---------------------------------------------------------------------- #
    cfg.llm = CN()

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_llm_cfg)


def assert_llm_cfg(cfg):
    ...


register_config("llm", extend_llm_cfg)
