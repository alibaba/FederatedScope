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

    # language model for hetero NLP tasks
    cfg.model.stage = ''  # ['assign', 'contrast']
    cfg.model.model_type = 'google/bert_uncased_L-2_H-128_A-2'
    cfg.model.pretrain_tasks = []
    cfg.model.downstream_tasks = []
    cfg.model.num_labels = 1
    cfg.model.max_length = 200
    cfg.model.min_length = 1
    cfg.model.no_repeat_ngram_size = 3
    cfg.model.length_penalty = 2.0
    cfg.model.num_beams = 5
    cfg.model.label_smoothing = 0.1
    cfg.model.n_best_size = 20
    cfg.model.max_answer_len = 30
    cfg.model.null_score_diff_threshold = 0.0
    cfg.model.use_contrastive_loss = False
    cfg.model.contrast_topk = 100
    cfg.model.contrast_temp = 1.0

    # prompt learning
    cfg.model.server_prefix_len = 0
    cfg.model.client_prefix_len = 0
    cfg.model.num_server_layers = 0
    cfg.model.num_client_layers = 0
    cfg.model.server_freeze_param = []
    cfg.model.client_freeze_param = []
    cfg.model.alter_model_param = []
    cfg.model.alter_prompt_param = []
    cfg.model.use_fp16 = False
    cfg.model.use_c2s_kd_loss = False
    cfg.model.use_s2c_kd_loss = False
    cfg.model.only_use_hidden_loss = False
    cfg.model.share_server_layer_param = False
    cfg.model.share_client_layer_param = False
    cfg.model.num_server_layers_per_cell = 1
    cfg.model.num_client_layers_per_cell = 1

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
