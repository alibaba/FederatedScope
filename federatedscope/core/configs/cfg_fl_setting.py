import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config
import torch

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
    cfg.federate.merge_test_data = False  # For efficient simulation, users
    # can choose to merge the test data and perform global evaluation,
    # instead of perform test at each client
    cfg.federate.merge_val_data = False  # Enabled only when
    # `merge_test_data` is True, also for efficient simulation

    # the method name is used to internally determine composition of
    # different aggregators, messages, handlers, etc.,
    cfg.federate.method = "FedAvg"
    cfg.federate.ignore_weight = False
    cfg.federate.use_ss = False  # Whether to apply Secret Sharing
    cfg.federate.restore_from = ''
    cfg.federate.save_to = ''
    cfg.federate.join_in_info = [
    ]  # The information requirements (from server) for join_in
    cfg.federate.sampler = 'uniform'  # the strategy for sampling client
    # in each training round, ['uniform', 'group']
    cfg.federate.resource_info_file = ""  # the device information file to
    # record computation and communication ability

    # The configurations for parallel in standalone
    cfg.federate.process_num = 1
    cfg.federate.master_addr = '127.0.0.1'  # parameter of torch distributed
    cfg.federate.master_port = 29500  # parameter of torch distributed

    # atc (TODO: merge later)
    cfg.federate.atc_vanilla = False
    cfg.federate.atc_load_from = ''

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
    cfg.distribute.data_idx = -1  # data_idx is used to specify the data
    # index in distributed mode when adopting a centralized dataset for
    # simulation (formatted as {data_idx: data/dataloader}).
    # data_idx = -1 means that the whole dataset is owned by the participant.
    # when data_idx is other invalid values excepted for -1, we randomly
    # sample the data_idx for simulation
    cfg.distribute.grpc_max_send_message_length = 300 * 1024 * 1024  # 300M
    cfg.distribute.grpc_max_receive_message_length = 300 * 1024 * 1024  # 300M
    cfg.distribute.grpc_enable_http_proxy = False
    cfg.distribute.grpc_compression = 'nocompression'  # [deflate, gzip]

    # ---------------------------------------------------------------------- #
    # Vertical FL related options (for demo)
    # ---------------------------------------------------------------------- #
    cfg.vertical = CN()
    cfg.vertical.use = False
    cfg.vertical.mode = 'feature_gathering'
    # ['feature_gathering', 'label_scattering']
    cfg.vertical.dims = [5, 10]  # Client 1 has the first 5 features,
    # and Client 2 has the last 5 features
    cfg.vertical.encryption = 'paillier'
    cfg.vertical.key_size = 3072
    cfg.vertical.algo = 'lr'  # ['lr', 'xgb', 'gbdt', 'rf']
    cfg.vertical.feature_subsample_ratio = 1.0
    cfg.vertical.protect_object = ''  # [feature_order, grad_and_hess]
    cfg.vertical.protect_method = ''
    # [dp, op_boost] for protect_object = feature_order
    # [he] for protect_object = grad_and_hess
    cfg.vertical.protect_args = []
    # Default values for 'dp': {'bucket_num':100, 'epsilon':None}
    # Default values for 'op_boost': {'algo':'global', 'lower_bound':1,
    #                                 'upper_bound':100, 'epsilon':2}
    cfg.vertical.eval_protection = ''  # ['', 'he']
    cfg.vertical.data_size_for_debug = 0  # use a subset for debug in vfl,
    # 0 indicates using the entire dataset (disable debug mode)

    cfg.adapter = CN()
    cfg.adapter.use = False
    cfg.adapter.args = []

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

    assert not cfg.federate.merge_test_data or (
            cfg.federate.merge_test_data and cfg.federate.mode == 'standalone'
    ), "The operation of merging test data can only used in standalone for " \
       "efficient simulation, please change 'federate.merge_test_data' to " \
       "False or change 'federate.mode' to 'distributed'."
    if cfg.federate.merge_test_data and not cfg.federate.make_global_eval:
        cfg.federate.make_global_eval = True
        logger.warning('Set cfg.federate.make_global_eval=True since '
                       'cfg.federate.merge_test_data=True')

    if cfg.federate.process_num > 1 and cfg.federate.mode != 'standalone':
        cfg.federate.process_num = 1
        logger.warning('Parallel training can only be used in standalone mode'
                       ', thus cfg.federate.process_num is modified to 1')
    if cfg.federate.process_num > 1 and not torch.cuda.is_available():
        cfg.federate.process_num = 1
        logger.warning(
            'No GPU found for your device, set cfg.federate.process_num=1')
    if torch.cuda.device_count() < cfg.federate.process_num:
        cfg.federate.process_num = torch.cuda.device_count()
        logger.warning(
            'We found the number of gpu is insufficient, '
            f'thus cfg.federate.process_num={cfg.federate.process_num}')
    # TODO
    if cfg.vertical.use:
        if cfg.vertical.algo == 'lr' and hasattr(cfg, "trainer") and \
                cfg.trainer.type != 'none':
            logger.warning(f"When given cfg.vertical.algo = 'lr', the value "
                           f"of cfg.trainer.type is expected to be 'none' "
                           f"but got {cfg.trainer.type}. Therefore "
                           f"cfg.trainer.type is changed to 'none' here")
            cfg.trainer.type = 'none'
        if cfg.vertical.algo == 'lr' and hasattr(cfg, "model") and \
                cfg.model.type != 'lr':
            logger.warning(f"When given cfg.vertical.algo = 'lr', the value "
                           f"of cfg.model.type is expected to be 'lr' "
                           f"but got {cfg.model.type}. Therefore "
                           f"cfg.model.type is changed to 'lr' here")
            cfg.model.type = 'lr'
        if cfg.vertical.algo in ['xgb', 'gbdt'] and hasattr(cfg, "trainer") \
                and cfg.trainer.type.lower() != 'verticaltrainer':
            logger.warning(
                f"When given cfg.vertical.algo = 'xgb' or 'gbdt', the value "
                f"of cfg.trainer.type is expected to be "
                f"'verticaltrainer' but got {cfg.trainer.type}. "
                f"Therefore cfg.trainer.type is changed to "
                f"'verticaltrainer' here")
            cfg.trainer.type = 'verticaltrainer'
        if cfg.vertical.algo == 'xgb' and hasattr(cfg, "model") and \
                cfg.model.type != 'xgb_tree':
            logger.warning(f"When given cfg.vertical.algo = 'xgb', the value "
                           f"of cfg.model.type is expected to be 'xgb_tree' "
                           f"but got {cfg.model.type}. Therefore "
                           f"cfg.model.type is changed to 'xgb_tree' here")
            cfg.model.type = 'xgb_tree'
        elif cfg.vertical.algo == 'gbdt' and hasattr(cfg, "model") and \
                cfg.model.type != 'gbdt_tree':
            logger.warning(f"When given cfg.vertical.algo = 'gbdt', the value "
                           f"of cfg.model.type is expected to be 'gbdt_tree' "
                           f"but got {cfg.model.type}. Therefore "
                           f"cfg.model.type is changed to 'gbdt_tree' here")
            cfg.model.type = 'gbdt_tree'

        if not (cfg.vertical.feature_subsample_ratio > 0
                and cfg.vertical.feature_subsample_ratio <= 1.0):
            raise ValueError(f'The value of vertical.feature_subsample_ratio '
                             f'must be in (0, 1.0], but got '
                             f'{cfg.vertical.feature_subsample_ratio}')

    if cfg.distribute.use and cfg.distribute.grpc_compression.lower() not in [
            'nocompression', 'deflate', 'gzip'
    ]:
        raise ValueError(f'The type of grpc compression is expected to be one '
                         f'of ["nocompression", "deflate", "gzip"], but got '
                         f'{cfg.distribute.grpc_compression}.')


register_config("fl_setting", extend_fl_setting_cfg)
