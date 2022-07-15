import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

logger = logging.getLogger(__name__)


def extend_fl_setting_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # Federate learning related options
    # ---------------------------------------------------------------------- #
    cfg.federate = CN()

    cfg.federate.client_num = 0
    cfg.federate.sample_client_num = -1
    cfg.federate.sample_client_rate = -1.0
    cfg.federate.unseen_clients_rate = 0.0
    cfg.federate.total_round_num = 50
    cfg.federate.mode = 'standalone'
    cfg.federate.share_local_model = False
    cfg.federate.data_weighted_aggr = False  # If True, the weight of aggr is
    # the number of training samples in dataset.
    cfg.federate.online_aggr = False
    cfg.federate.make_global_eval = False
    cfg.federate.use_diff = False

    # the method name is used to internally determine composition of
    # different aggregators, messages, handlers, etc.,
    cfg.federate.method = "FedAvg"
    cfg.federate.ignore_weight = False
    cfg.federate.use_ss = False  # Whether to apply Secret Sharing
    cfg.federate.restore_from = ''
    cfg.federate.save_to = ''
    cfg.federate.join_in_info = [
    ]  # The information requirements (from server) for join_in
    cfg.federate.sampler = 'uniform'  # the strategy for sampling client in
    # each training round, ['uniform', 'group']

    # ---------------------------------------------------------------------- #
    # Distribute training related options
    # ---------------------------------------------------------------------- #
    cfg.distribute = CN()

    cfg.distribute.use = False
    cfg.distribute.server_host = '0.0.0.0'
    cfg.distribute.server_port = 50050
    cfg.distribute.client_host = '0.0.0.0'
    cfg.distribute.client_port = 50050
    cfg.distribute.role = 'client'
    cfg.distribute.data_file = 'data'
    cfg.distribute.data_idx = -1
    cfg.distribute.grpc_max_send_message_length = 100 * 1024 * 1024
    cfg.distribute.grpc_max_receive_message_length = 100 * 1024 * 1024
    cfg.distribute.grpc_enable_http_proxy = False

    # ---------------------------------------------------------------------- #
    # Vertical FL related options (for demo)
    # ---------------------------------------------------------------------- #
    cfg.vertical = CN()
    cfg.vertical.use = False
    cfg.vertical.encryption = 'paillier'
    cfg.vertical.dims = [5, 10]
    cfg.vertical.key_size = 3072

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_fl_setting_cfg)


def assert_fl_setting_cfg(cfg):
    assert cfg.federate.mode in ["standalone", "distributed"], \
        f"Please specify the cfg.federate.mode as the string standalone or " \
        f"distributed. But got {cfg.federate.mode}."

    # =============  client num related  ==============
    assert not (cfg.federate.client_num == 0
                and cfg.federate.mode == 'distributed'
                ), "Please configure the cfg.federate. in distributed mode. "

    assert 0 <= cfg.federate.unseen_clients_rate < 1, \
        "You specified in-valid cfg.federate.unseen_clients_rate"
    if 0 < cfg.federate.unseen_clients_rate < 1 and cfg.federate.method in [
            "local", "global"
    ]:
        logger.warning(
            "In local/global training mode, the unseen_clients_rate is "
            "in-valid, plz check your config")
        unseen_clients_rate = 0.0
        cfg.federate.unseen_clients_rate = unseen_clients_rate
    else:
        unseen_clients_rate = cfg.federate.unseen_clients_rate
    participated_client_num = max(
        1, int((1 - unseen_clients_rate) * cfg.federate.client_num))

    # sample client num pre-process
    sample_client_num_valid = (
        0 < cfg.federate.sample_client_num <=
        cfg.federate.client_num) and cfg.federate.client_num != 0
    sample_client_rate_valid = (0 < cfg.federate.sample_client_rate <= 1)

    sample_cfg_valid = sample_client_rate_valid or sample_client_num_valid
    non_sample_case = cfg.federate.method in ["local", "global"]
    if non_sample_case and sample_cfg_valid:
        logger.warning("In local/global training mode, "
                       "the sampling related configs are in-valid, "
                       "we will use all clients. ")

    if cfg.federate.method == "global":
        logger.info(
            "In global training mode, we will put all data in a proxy client. "
        )
        if cfg.federate.make_global_eval:
            cfg.federate.make_global_eval = False
            logger.warning(
                "In global training mode, we will conduct global evaluation "
                "in a proxy client rather than the server. The configuration "
                "cfg.federate.make_global_eval will be False.")

    if non_sample_case or not sample_cfg_valid:
        # (a) use all clients
        # in standalone mode, federate.client_num may be modified from 0 to
        # num_of_all_clients after loading the data
        if cfg.federate.client_num != 0:
            cfg.federate.sample_client_num = participated_client_num
    else:
        # (b) sampling case
        if sample_client_rate_valid:
            # (b.1) use sample_client_rate
            old_sample_client_num = cfg.federate.sample_client_num
            cfg.federate.sample_client_num = max(
                1,
                int(cfg.federate.sample_client_rate * participated_client_num))
            if sample_client_num_valid:
                logger.warning(
                    f"Users specify both valid sample_client_rate as"
                    f" {cfg.federate.sample_client_rate} "
                    f"and sample_client_num as {old_sample_client_num}.\n"
                    f"\t\tWe will use the sample_client_rate value to "
                    f"calculate "
                    f"the actual number of participated clients as"
                    f" {cfg.federate.sample_client_num}.")
            # (b.2) use sample_client_num, commented since the below two
            # lines do not change anything
            # elif sample_client_num_valid:
            #     cfg.federate.sample_client_num = \
            #     cfg.federate.sample_client_num

    if cfg.federate.use_ss:
        assert cfg.federate.client_num == cfg.federate.sample_client_num, \
            "Currently, we support secret sharing only in " \
            "all-client-participation case"

        assert cfg.federate.method != "local", \
            "Secret sharing is not supported in local training mode"

    # =============   aggregator related   ================
    assert (not cfg.federate.online_aggr) or (
        not cfg.federate.use_ss
    ), "Have not supported to use online aggregator and secrete sharing at " \
       "the same time"


register_config("fl_setting", extend_fl_setting_cfg)
