import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

logger = logging.getLogger(__name__)


def extend_llm_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # LLM related options
    # ---------------------------------------------------------------------- #
    cfg.llm = CN()
    cfg.llm.tok_len = 128

    cfg.llm.chat = CN()
    cfg.llm.chat.max_history_len = 10
    cfg.llm.chat.max_len = 100

    # ---------------------------------------------------------------------- #
    # Adapters for LLM
    # ---------------------------------------------------------------------- #
    cfg.llm.adapter = CN()
    cfg.llm.adapter.use = False
    cfg.llm.adapter.args = []


def assert_llm_cfg(cfg):
    pass


register_config("llm", extend_llm_cfg)
