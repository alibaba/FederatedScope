from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_textdt_cfg(cfg):
    cfg.federate.load_from = None

    cfg.model.bert_type = None
    cfg.model.num_labels = CN()
    cfg.model.num_labels.sts = None
    cfg.model.num_labels.imdb = None
    cfg.model.num_labels.squad = None
    cfg.model.label_smoothing = None
    cfg.model.maml = None

    cfg.eval.n_best_size = None
    cfg.eval.max_answer_len = None
    cfg.eval.null_score_diff_threshold = None

    cfg.data.dir = CN()
    cfg.data.dir.sts = None
    cfg.data.dir.imdb = None
    cfg.data.dir.squad = None
    cfg.data.max_seq_len = CN()
    cfg.data.max_seq_len.sts = None
    cfg.data.max_seq_len.imdb = None
    cfg.data.max_seq_len.squad = None
    cfg.data.max_tgt_len = None
    cfg.data.max_query_len = None
    cfg.data.trunc_stride = None
    cfg.data.cache_dir = None

    cfg.scheduler = CN()
    cfg.scheduler.type = None
    cfg.scheduler.warmup_ratio = None

    cfg.trainer.disp_freq = None
    cfg.trainer.val_freq = None
    cfg.trainer.grad_accum_count = None
    cfg.trainer.train_steps = None
    cfg.trainer.test_only = None

    cfg.maml = CN()
    cfg.maml.inner_lr = None

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_textdt_cfg)


def assert_textdt_cfg(cfg):
    pass


register_config('text-dt', extend_textdt_cfg)
