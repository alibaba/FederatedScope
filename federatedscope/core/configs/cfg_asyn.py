import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_asyn_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # Asynchronous related options
    # ---------------------------------------------------------------------- #
    cfg.asyn = CN()

    cfg.asyn.use = False
    cfg.asyn.time_budget = 0
    cfg.asyn.min_received_num = 2
    cfg.asyn.min_received_rate = -1.0
    cfg.asyn.staleness_toleration = 0
    cfg.asyn.staleness_discount_factor = 1.0
    cfg.asyn.aggregator = 'goal_achieved'  # ['goal_achieved', 'time_up']
    # 'goal_achieved': perform aggregation when the defined number of feedback
    # has been received; 'time_up': perform aggregation when the allocated
    # time budget has been run out
    cfg.asyn.broadcast_manner = 'after_aggregating'  # ['after_aggregating',
    # 'after_receiving'] 'after_aggregating': broadcast the up-to-date global
    # model after performing federated aggregation;
    # 'after_receiving': broadcast the up-to-date global model after receiving
    # the model update from clients
    cfg.asyn.overselection = False

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_asyn_cfg)


def assert_asyn_cfg(cfg):
    if not cfg.asyn.use:
        return True
    # to ensure a valid time budget
    assert isinstance(cfg.asyn.time_budget, int) or isinstance(
        cfg.asyn.time_budget, float
    ), "The time budget (seconds) must be an int or a float value, " \
       "but {} is got".format(
        type(cfg.asyn.time_budget))

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

    # to ensure a valid staleness toleation
    assert cfg.asyn.staleness_toleration >= 0 and isinstance(
        cfg.asyn.staleness_toleration, int
    ), f"Please provide a valid staleness toleration value, " \
       f"expect an integer value that is larger or equal to 0, " \
       f"but got {cfg.asyn.staleness_toleration}."

    assert cfg.asyn.aggregator in ["goal_achieved", "time_up"], \
        f"Please specify the cfg.asyn.aggregator as string 'goal_achieved' " \
        f"or 'time_up'. But got {cfg.asyn.aggregator}."
    assert cfg.asyn.broadcast_manner in ["after_aggregating",
                                         "after_receiving"], \
        f"Please specify the cfg.asyn.broadcast_manner as the string " \
        f"'after_aggregating' or 'after_receiving'. " \
        f"But got {cfg.asyn.broadcast_manner}."


register_config("asyn", extend_asyn_cfg)
