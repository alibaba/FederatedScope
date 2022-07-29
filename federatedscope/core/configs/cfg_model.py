from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_model_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # Model related options
    # ---------------------------------------------------------------------- #
    cfg.model = CN()

    cfg.model.model_num_per_trainer = 1  # some methods may leverage more
    # than one model in each trainer
    cfg.model.type = 'lr'
    cfg.model.use_bias = True
    cfg.model.task = 'node'
    cfg.model.hidden = 256
    cfg.model.dropout = 0.5
    cfg.model.in_channels = 0  # If 0, model will be built by data.shape
    cfg.model.out_channels = 1
    cfg.model.layer = 2  # In GPR-GNN, K = layer
    cfg.model.graph_pooling = 'mean'
    cfg.model.embed_size = 8
    cfg.model.num_item = 0
    cfg.model.num_user = 0
    cfg.model.input_shape = ()  # A tuple, e.g., (in_channel, h, w)

    # ---------------------------------------------------------------------- #
    # Criterion related options
    # ---------------------------------------------------------------------- #
    cfg.criterion = CN()

    cfg.criterion.type = 'MSELoss'

    # ---------------------------------------------------------------------- #
    # regularizer related options
    # ---------------------------------------------------------------------- #
    cfg.regularizer = CN()

    cfg.regularizer.type = ''
    cfg.regularizer.mu = 0.

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_model_cfg)


def assert_model_cfg(cfg):
    pass


register_config("model", extend_model_cfg)
