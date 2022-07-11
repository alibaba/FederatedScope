from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_fl_algo_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # fedopt related options, general fl
    # ---------------------------------------------------------------------- #
    cfg.fedopt = CN()

    cfg.fedopt.use = False

    cfg.fedopt.optimizer = CN(new_allowed=True)
    cfg.fedopt.optimizer.type = 'SGD'
    cfg.fedopt.optimizer.lr = 0.01

    # ---------------------------------------------------------------------- #
    # fedprox related options, general fl
    # ---------------------------------------------------------------------- #
    cfg.fedprox = CN()

    cfg.fedprox.use = False
    cfg.fedprox.mu = 0.

    # ---------------------------------------------------------------------- #
    # Personalization related options, pFL
    # ---------------------------------------------------------------------- #
    cfg.personalization = CN()

    # client-distinct param names, e.g., ['pre', 'post']
    cfg.personalization.local_param = []
    cfg.personalization.share_non_trainable_para = False
    cfg.personalization.local_update_steps = -1
    # @regular_weight:
    # The smaller the regular_weight is, the stronger emphasising on
    # personalized model
    # For Ditto, the default value=0.1, the search space is [0.05, 0.1, 0.2,
    # 1, 2]
    # For pFedMe, the default value=15
    cfg.personalization.regular_weight = 0.1

    # @lr:
    # 1) For pFedME, the personalized learning rate to calculate theta
    # approximately using K steps
    # 2) 0.0 indicates use the value according to optimizer.lr in case of
    # users have not specify a valid lr
    cfg.personalization.lr = 0.0

    cfg.personalization.K = 5  # the local approximation steps for pFedMe
    cfg.personalization.beta = 1.0  # the average moving parameter for pFedMe

    # ---------------------------------------------------------------------- #
    # FedSage+ related options, gfl
    # ---------------------------------------------------------------------- #
    cfg.fedsageplus = CN()

    cfg.fedsageplus.num_pred = 5
    cfg.fedsageplus.gen_hidden = 128
    cfg.fedsageplus.hide_portion = 0.5
    cfg.fedsageplus.fedgen_epoch = 200
    cfg.fedsageplus.loc_epoch = 1
    cfg.fedsageplus.a = 1.0
    cfg.fedsageplus.b = 1.0
    cfg.fedsageplus.c = 1.0

    # ---------------------------------------------------------------------- #
    # GCFL+ related options, gfl
    # ---------------------------------------------------------------------- #
    cfg.gcflplus = CN()

    cfg.gcflplus.EPS_1 = 0.05
    cfg.gcflplus.EPS_2 = 0.1
    cfg.gcflplus.seq_length = 5
    cfg.gcflplus.standardize = False

    # ---------------------------------------------------------------------- #
    # FLIT+ related options, gfl
    # ---------------------------------------------------------------------- #
    cfg.flitplus = CN()

    cfg.flitplus.tmpFed = 0.5  # gamma in focal loss (Eq.4)
    cfg.flitplus.lambdavat = 0.5  # lambda in phi (Eq.10)
    cfg.flitplus.factor_ema = 0.8  # beta in omega (Eq.12)
    cfg.flitplus.weightReg = 1.0  # balance lossLocalLabel and lossLocalVAT

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_fl_algo_cfg)


def assert_fl_algo_cfg(cfg):
    if cfg.personalization.local_update_steps == -1:
        # By default, use the same step to normal mode
        cfg.personalization.local_update_steps = \
            cfg.train.local_update_steps
        cfg.personalization.local_update_steps = \
            cfg.train.local_update_steps

    if cfg.personalization.lr <= 0.0:
        # By default, use the same lr to normal mode
        cfg.personalization.lr = cfg.train.optimizer.lr


register_config("fl_algo", extend_fl_algo_cfg)
