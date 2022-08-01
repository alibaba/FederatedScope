import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_asyn_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # Asynchronous related options
    # ---------------------------------------------------------------------- #
    cfg.asyn = CN()

    cfg.asyn.use = True
    cfg.asyn.timeout = 0
    cfg.asyn.min_received_num = 2
    cfg.asyn.min_received_rate = -1.0

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_asyn_cfg)


def assert_asyn_cfg(cfg):
    # to ensure a valid timeout seconds
    assert isinstance(cfg.asyn.timeout, int) or isinstance(
        cfg.asyn.timeout, float
    ), "The timeout (seconds) must be an int or a float value, but {} is " \
       "got".format(
        type(cfg.asyn.timeout))

    # min received num pre-process
    min_received_num_valid = (0 < cfg.asyn.min_received_num <=
                              cfg.federate.sample_client_num)
    min_received_rate_valid = (0 < cfg.asyn.min_received_rate <= 1)
    # (a) sampling case
    if min_received_rate_valid:
        # (a.1) use min_received_rate
        old_min_received_num = cfg.asyn.min_received_num
        cfg.asyn.min_received_num = max(
            1,
            int(cfg.asyn.min_received_rate * cfg.federate.sample_client_num))
        if min_received_num_valid:
            logging.warning(
                f"Users specify both valid min_received_rate as"
                f" {cfg.asyn.min_received_rate} "
                f"and min_received_num as {old_min_received_num}.\n"
                f"\t\tWe will use the min_received_rate value to calculate "
                f"the actual number of participated clients as"
                f" {cfg.asyn.min_received_num}.")
    # (a.2) use min_received_num, commented since the below two lines do not
    # change anything elif min_received_rate:
    #     cfg.asyn.min_received_num = cfg.asyn.min_received_num
    if not (min_received_num_valid or min_received_rate_valid):
        # (b) non-sampling case, use all clients
        cfg.asyn.min_received_num = cfg.federate.sample_client_num


register_config("asyn", extend_asyn_cfg)
