import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

logger = logging.getLogger(__name__)


def extend_computation_quantization_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # quantization (for memory/computation efficiency) related options
    # ---------------------------------------------------------------------- #
    cfg.computation_quantization = CN()

    # Params
    # ['qlora', 'uniform']
    cfg.computation_quantization.method = 'none'
    cfg.computation_quantization.nbits = 4  # [4,8,16]

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_quant_cfg)


def assert_quant_cfg(cfg):

    if cfg.quantization.method.lower() not in ['none', 'qlora']:
        logger.warning(
            'Quantization for Communication method is expected '
            'to be one of ["none","qlora"]',
            f'but got "{cfg.quantization.method}". So we',
            'change it to "none"')

    if cfg.quantization.method.lower(
    ) != 'none' and cfg.quantization.nbits not in [4, 8, 16]:
        raise ValueError(f'The value of cfg.quantization.nbits is invalid, '
                         f'which is expected to be one on [4, 8, 16] but got '
                         f'{cfg.quantization.nbits}.')


register_config("computation_quantization",
                extend_computation_quantization_cfg)
