from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_textdt_cfg(cfg):
    cfg.federate.load_from = ''

    cfg.model.bert_type = 'bert-base-uncased'
    cfg.model.num_labels = CN()
    cfg.model.num_labels.sts = 1
    cfg.model.num_labels.imdb = 2
    cfg.model.num_labels.squad = 2
    cfg.model.label_smoothing = 0.0
    cfg.model.maml = False

    cfg.eval.n_best_size = 20
    cfg.eval.max_answer_len = 30
    cfg.eval.null_score_diff_threshold = 0.0

    cfg.data.dir = CN()
    cfg.data.dir.sts = 'data/STS-B'
    cfg.data.dir.imdb = 'data/imdb'
    cfg.data.dir.squad = 'data/squad2.0'
    cfg.data.max_seq_len = CN()
    cfg.data.max_seq_len.sts = 128
    cfg.data.max_seq_len.imdb = 128
    cfg.data.max_seq_len.squad = 128
    cfg.data.max_query_len = 64
    cfg.data.trunc_stride = 32
    cfg.data.cache_dir = ''
    cfg.data.debug = False

    cfg.scheduler = CN()
    cfg.scheduler.type = 'step'
    cfg.scheduler.warmup_ratio = 0.0

    cfg.trainer.disp_freq = 50
    cfg.trainer.val_freq = 100000000
    cfg.trainer.grad_accum_count = 1
    cfg.trainer.train_steps = 100

    cfg.maml = CN()
    cfg.maml.use = False
    cfg.maml.inner_lr = 1e-3

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_textdt_cfg)


def assert_textdt_cfg(cfg):
    pass


register_config('text-dt', extend_textdt_cfg)
