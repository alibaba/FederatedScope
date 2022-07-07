from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_data_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # Dataset related options
    # ---------------------------------------------------------------------- #
    cfg.data = CN()

    cfg.data.root = 'data'
    cfg.data.type = 'toy'
    cfg.data.args = []  # args for external dataset, eg. [{'download': True}]
    cfg.data.splitter = ''
    cfg.data.splitter_args = []  # args for splitter, eg. [{'alpha': 0.5}]
    cfg.data.transform = [
    ]  # transform for x, eg. [['ToTensor'], ['Normalize', {'mean': [
    # 0.1307], 'std': [0.3081]}]]
    cfg.data.target_transform = []  # target_transform for y, use as above
    cfg.data.pre_transform = [
    ]  # pre_transform for `torch_geometric` dataset, use as above
    cfg.data.batch_size = 64
    cfg.data.drop_last = False
    cfg.data.sizes = [10, 5]
    cfg.data.shuffle = True
    cfg.data.subsample = 1.0
    cfg.data.splits = [0.8, 0.1, 0.1]  # Train, valid, test splits
    cfg.data.consistent_label_distribution = False  # If True, the label
    # distributions of train/val/test set over clients will be kept
    # consistent during splitting
    cfg.data.cSBM_phi = [0.5, 0.5, 0.5]
    cfg.data.loader = ''
    cfg.data.num_workers = 0
    cfg.data.graphsaint = CN()
    cfg.data.graphsaint.walk_length = 2
    cfg.data.graphsaint.num_steps = 30

    # quadratic
    cfg.data.quadratic = CN()
    cfg.data.quadratic.dim = 1
    cfg.data.quadratic.min_curv = 0.02
    cfg.data.quadratic.max_curv = 12.5

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_data_cfg)


def assert_data_cfg(cfg):
    if cfg.data.loader == 'graphsaint-rw':
        assert cfg.model.layer == cfg.data.graphsaint.walk_length, 'Sample ' \
                                                                   'size ' \
                                                                   'mismatch'
    if cfg.data.loader == 'neighbor':
        assert cfg.model.layer == len(cfg.data.sizes), 'Sample size mismatch'
    if '@' in cfg.data.type:
        assert cfg.federate.client_num > 0, '`federate.client_num` should ' \
                                            'be greater than 0 when using ' \
                                            'external data'
        assert cfg.data.splitter, '`data.splitter` should not be empty when ' \
                                  'using external data'


register_config("data", extend_data_cfg)
