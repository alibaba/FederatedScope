from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_dp_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # nbafl(dp) related options
    # ---------------------------------------------------------------------- #
    cfg.nbafl = CN()

    # Params
    cfg.nbafl.use = False
    cfg.nbafl.mu = 0.
    cfg.nbafl.epsilon = 100.
    cfg.nbafl.w_clip = 1.
    cfg.nbafl.constant = 30.

    # ---------------------------------------------------------------------- #
    # VFL-SGDMF(dp) related options
    # ---------------------------------------------------------------------- #
    cfg.sgdmf = CN()

    cfg.sgdmf.use = False  # if use sgdmf algorithm
    cfg.sgdmf.R = 5.  # The upper bound of rating
    cfg.sgdmf.epsilon = 4.  # \epsilon in dp
    cfg.sgdmf.delta = 0.5  # \delta in dp
    cfg.sgdmf.constant = 1.  # constant
    cfg.sgdmf.theta = -1  # -1 means per-rating privacy, otherwise per-user
    # privacy

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_dp_cfg)


def assert_dp_cfg(cfg):
    pass


register_config("dp", extend_dp_cfg)
