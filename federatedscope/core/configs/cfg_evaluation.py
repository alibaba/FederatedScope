from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_evaluation_cfg(cfg):

    # ------------------------------------------------------------------------ #
    # Evaluation related options
    # ------------------------------------------------------------------------ #
    cfg.eval = CN()

    cfg.eval.save_data = False
    cfg.eval.freq = 1
    cfg.eval.metrics = []
    cfg.eval.report = ['weighted_avg', 'avg', 'fairness',
                       'raw']  # by default, we report comprehensive results
    cfg.eval.best_res_update_round_wise_key = "val_loss"

    # Monitoring, e.g., 'dissim' for B-local dissimilarity
    cfg.eval.monitoring = []

    # ------------------------------------------------------------------------ #
    # wandb related options
    # ------------------------------------------------------------------------ #
    cfg.wandb = CN()
    cfg.wandb.use = False
    cfg.wandb.name_user = ''
    cfg.wandb.name_project = ''

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_evaluation_cfg)


def assert_evaluation_cfg(cfg):
    pass


register_config("eval", extend_evaluation_cfg)
