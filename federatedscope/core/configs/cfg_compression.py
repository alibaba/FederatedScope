import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

logger = logging.getLogger(__name__)


def extend_compression_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # Compression (for communication efficiency) related options
    # ---------------------------------------------------------------------- #
    cfg.quantization = CN()

    # Params
    cfg.quantization.method = 'none'  # ['none', 'uniform']
    cfg.quantization.nbits = 8  # [8,16]

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_compression_cfg)


def assert_compression_cfg(cfg):

    if cfg.quantization.method.lower() not in ['none', 'uniform']:
        logger.warning(
            f'Quantization method is expected to be one of ["none",'
            f'"uniform"], but got "{cfg.quantization.method}". So we '
            f'change it to "none"')

    if cfg.quantization.method.lower(
    ) != 'none' and cfg.quantization.nbits not in [8, 16]:
        raise ValueError(f'The value of cfg.quantization.nbits is invalid, '
                         f'which is expected to be one on [8, 16] but got '
                         f'{cfg.quantization.nbits}.')


register_config("compression", extend_compression_cfg)
