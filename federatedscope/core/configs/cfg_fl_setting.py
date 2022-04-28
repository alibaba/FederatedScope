import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

logger = logging.getLogger(__name__)


def extend_fl_setting_cfg(cfg):
    # ------------------------------------------------------------------------ #
    # Federate learning related options
    # ------------------------------------------------------------------------ #
    cfg.federate = CN()

    cfg.federate.client_num = 0
    cfg.federate.sample_client_num = -1
    cfg.federate.sample_client_rate = -1.0
    cfg.federate.total_round_num = 50
    cfg.federate.mode = 'standalone'
    cfg.federate.local_update_steps = 1  # If the mode is `local`, `local_update_steps` is the epochs.
    cfg.federate.batch_or_epoch = 'batch'
    cfg.federate.share_local_model = False
    cfg.federate.data_weighted_aggr = False  # If True, the weight of aggr is the number of training samples in dataset.
    cfg.federate.online_aggr = False
    cfg.federate.make_global_eval = False

    # the method name is used to internally determine composition of different aggregators, messages, handlers, etc.,
    cfg.federate.method = "FedAvg"
    cfg.federate.ignore_weight = False
    cfg.federate.use_ss = False  # Whether to apply Secret Sharing
    cfg.federate.restore_from = ''
    cfg.federate.save_to = ''
    cfg.federate.join_in_info = [
    ]  # The information requirements (from server) for join_in

    # ------------------------------------------------------------------------ #
    # Distribute training related options
    # ------------------------------------------------------------------------ #
    cfg.distribute = CN()

    cfg.distribute.use = False
    cfg.distribute.server_host = '0.0.0.0'
    cfg.distribute.server_port = 50050
    cfg.distribute.client_host = '0.0.0.0'
    cfg.distribute.client_port = 50050
    cfg.distribute.role = 'client'
    cfg.distribute.data_file = 'data'
    cfg.distribute.grpc_max_send_message_length = 100 * 1024 * 1024
    cfg.distribute.grpc_max_receive_message_length = 100 * 1024 * 1024
    cfg.distribute.grpc_enable_http_proxy = False

    # ------------------------------------------------------------------------ #
    # Vertical FL related options (for demo)
    # ------------------------------------------------------------------------ #
    cfg.vertical = CN()
    cfg.vertical.use = False
    cfg.vertical.encryption = 'paillier'
    cfg.vertical.dims = [5, 10]
    cfg.vertical.key_size = 3072

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_fl_setting_cfg)


def assert_fl_setting_cfg(cfg):
    if cfg.federate.batch_or_epoch not in ['batch', 'epoch']:
        raise ValueError(
            "Value of 'cfg.federate.batch_or_epoch' must be chosen from ['batch', 'epoch']."
        )

    assert cfg.federate.mode in ["standalone", "distributed", "local"], \
        f"Please specify the cfg.federate.mode as the string standalone, distributed or local. But got {cfg.federate.mode}."

    # client num related
    assert not (cfg.federate.client_num == 0
                and cfg.federate.mode == 'distributed'
                ), "Please configure the cfg.federate. in distributed mode. "

    # sample client num pre-process
    sample_client_num_valid = (0 < cfg.federate.sample_client_num <=
                               cfg.federate.client_num)
    sample_client_rate_valid = (0 < cfg.federate.sample_client_rate <= 1)
    # (a) sampling case
    if sample_client_rate_valid:
        # (a.1) use sample_client_rate
        old_sample_client_num = cfg.federate.sample_client_num
        cfg.federate.sample_client_num = max(
            1, int(cfg.federate.sample_client_rate * cfg.federate.client_num))
        if sample_client_num_valid:
            logger.warning(
                f"Users specify both valid sample_client_rate as {cfg.federate.sample_client_rate} "
                f"and sample_client_num as {old_sample_client_num}.\n"
                f"\t\tWe will use the sample_client_rate value to calculate "
                f"the actual number of participated clients as {cfg.federate.sample_client_num}."
            )
    # (a.2) use sample_client_num, commented since the below two lines do not change anything
    # elif sample_client_num_valid:
    #     cfg.federate.sample_client_num = cfg.federate.sample_client_num
    if not (sample_client_rate_valid or sample_client_num_valid):
        # (b) non-sampling case, use all clients
        cfg.federate.sample_client_num = cfg.federate.client_num

    if cfg.federate.use_ss:
        assert cfg.federate.client_num == cfg.federate.sample_client_num, \
            "Currently, we support secret sharing only in all-client-participation case"

    # aggregator related
    assert (not cfg.federate.online_aggr) or (
        not cfg.federate.use_ss
    ), "Have not supported to use online aggregator and secrete sharing at the same time"


register_config("fl_setting", extend_fl_setting_cfg)
