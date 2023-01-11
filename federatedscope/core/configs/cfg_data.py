import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

logger = logging.getLogger(__name__)


def extend_data_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # Dataset related options
    # ---------------------------------------------------------------------- #
    cfg.data = CN()

    cfg.data.root = 'data'
    cfg.data.type = 'toy'
    cfg.data.save_data = False  # whether to save the generated toy data
    cfg.data.args = []  # args for external dataset, eg. [{'download': True}]
    cfg.data.splitter = ''
    cfg.data.splitter_args = []  # args for splitter, eg. [{'alpha': 0.5}]
    cfg.data.server_holds_all = False  # whether the server (workers with
    # idx 0) holds all data, useful in global training/evaluation case
    cfg.data.subsample = 1.0
    cfg.data.splits = [0.8, 0.1, 0.1]  # Train, valid, test splits
    cfg.data.consistent_label_distribution = False  # If True, the label
    # distributions of train/val/test set over clients will be kept
    # consistent during splitting
    cfg.data.cSBM_phi = [0.5, 0.5, 0.5]

    cfg.data.transform = [
    ]  # transform for x, eg. [['ToTensor'], ['Normalize', {'mean': [
    # 0.1307], 'std': [0.3081]}]]
    cfg.data.target_transform = []  # target_transform for y, use as above
    cfg.data.pre_transform = [
    ]  # pre_transform for `torch_geometric` dataset, use as above

    # If not provided, use `cfg.data.transform` for all splits
    cfg.data.val_transform = []
    cfg.data.val_target_transform = []
    cfg.data.val_pre_transform = []
    cfg.data.test_transform = []
    cfg.data.test_target_transform = []
    cfg.data.test_pre_transform = []

    # DataLoader related args
    cfg.dataloader = CN()
    cfg.dataloader.type = 'base'
    cfg.dataloader.batch_size = 64
    cfg.dataloader.shuffle = True
    cfg.dataloader.num_workers = 0
    cfg.dataloader.drop_last = False
    cfg.dataloader.pin_memory = False
    # GFL: graphsaint DataLoader
    cfg.dataloader.walk_length = 2
    cfg.dataloader.num_steps = 30
    # GFL: neighbor sampler DataLoader
    cfg.dataloader.sizes = [10, 5]
    # DP: -1 means per-rating privacy, otherwise per-user privacy
    cfg.dataloader.theta = -1

    # quadratic
    cfg.data.quadratic = CN()
    cfg.data.quadratic.dim = 1
    cfg.data.quadratic.min_curv = 0.02
    cfg.data.quadratic.max_curv = 12.5

    # Hetero NLP tasks data (for ATC)
    cfg.data.hetero_data_name = []  # multiple datasets
    cfg.data.num_of_client_for_data = [
    ]  # each dataset can be splited into several clients
    cfg.data.max_seq_len = 384
    cfg.data.max_tgt_len = 128
    cfg.data.max_query_len = 128
    cfg.data.trunc_stride = 128
    cfg.data.cache_dir = ''
    cfg.data.hetero_synth_batch_size = 32
    cfg.data.hetero_synth_prim_weight = 0.5
    cfg.data.hetero_synth_feat_dim = 128
    cfg.data.num_contrast = 0
    cfg.data.is_debug = False

    # feature engineering
    cfg.feat_engr = CN()
    cfg.feat_engr.type = ''
    cfg.feat_engr.scenario = 'hfl'
    cfg.feat_engr.num_bins = 5  # Used for binning
    cfg.feat_engr.selec_threshold = 0.05  # Used for feature selection
    cfg.feat_engr.selec_woe_binning = 'quantile'

    cfg.feat_engr.secure = CN()
    cfg.feat_engr.secure.type = 'encrypt'
    cfg.feat_engr.secure.key_size = 3072

    cfg.feat_engr.secure.encrypt = CN()
    cfg.feat_engr.secure.encrypt.type = 'dummy'

    cfg.feat_engr.secure.dp = CN()  # under dev

    # --------------- outdated configs ---------------
    # TODO: delete this code block
    cfg.data.loader = ''
    cfg.data.batch_size = 64
    cfg.data.shuffle = True
    cfg.data.num_workers = 0
    cfg.data.drop_last = False
    cfg.data.walk_length = 2
    cfg.data.num_steps = 30
    cfg.data.sizes = [10, 5]

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_data_cfg)


def assert_data_cfg(cfg):
    if cfg.dataloader.type == 'graphsaint-rw':
        assert cfg.model.layer == cfg.dataloader.walk_length, 'Sample ' \
                                                                   'size ' \
                                                                   'mismatch'
    if cfg.dataloader.type == 'neighbor':
        assert cfg.model.layer == len(
            cfg.dataloader.sizes), 'Sample size mismatch'
    if '@' in cfg.data.type:
        assert cfg.federate.client_num > 0, '`federate.client_num` should ' \
                                            'be greater than 0 when using ' \
                                            'external data'
        assert cfg.data.splitter, '`data.splitter` should not be empty when ' \
                                  'using external data'

    # hetero NLP taks data
    if len(cfg.data.num_of_client_for_data) > 0:
        assert cfg.federate.client_num == \
               sum(cfg.data.num_of_client_for_data), '`federate.client_num` ' \
                                                     'should be equal to sum '\
                                                     'of `data.num_of_client'\
                                                     '_for_data`'

    # --------------------------------------------------------------------
    # For compatibility with older versions of FS
    # TODO: delete this code block
    if cfg.data.loader != '':
        logger.warning('config `cfg.data.loader` will be removed in the '
                       'future, use `cfg.dataloader.type` instead.')
        cfg.dataloader.type = cfg.data.loader
    if cfg.data.batch_size != 64:
        logger.warning('config `cfg.data.batch_size` will be removed in the '
                       'future, use `cfg.dataloader.batch_size` instead.')
        cfg.dataloader.batch_size = cfg.data.batch_size
    if not cfg.data.shuffle:
        logger.warning('config `cfg.data.shuffle` will be removed in the '
                       'future, use `cfg.dataloader.shuffle` instead.')
        cfg.dataloader.shuffle = cfg.data.shuffle
    if cfg.data.num_workers != 0:
        logger.warning('config `cfg.data.num_workers` will be removed in the '
                       'future, use `cfg.dataloader.num_workers` instead.')
        cfg.dataloader.num_workers = cfg.data.num_workers
    if cfg.data.drop_last:
        logger.warning('config `cfg.data.drop_last` will be removed in the '
                       'future, use `cfg.dataloader.drop_last` instead.')
        cfg.dataloader.drop_last = cfg.data.drop_last
    if cfg.data.walk_length != 2:
        logger.warning('config `cfg.data.walk_length` will be removed in the '
                       'future, use `cfg.dataloader.walk_length` instead.')
        cfg.dataloader.walk_length = cfg.data.walk_length
    if cfg.data.num_steps != 30:
        logger.warning('config `cfg.data.num_steps` will be removed in the '
                       'future, use `cfg.dataloader.num_steps` instead.')
        cfg.dataloader.num_steps = cfg.data.num_steps
    if cfg.data.sizes != [10, 5]:
        logger.warning('config `cfg.data.sizes` will be removed in the '
                       'future, use `cfg.dataloader.sizes` instead.')
        cfg.dataloader.sizes = cfg.data.sizes
    # --------------------------------------------------------------------


register_config("data", extend_data_cfg)
