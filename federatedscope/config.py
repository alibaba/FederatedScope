import logging
import os
from yacs.config import CfgNode as CN

# from graphgym.utils.io import makedirs_rm_exist

import federatedscope.register as register

logger = logging.getLogger(__name__)

# Global config object
cfg = CN()


def set_cfg(cfg):
    r'''
    This function sets the default config value.
    1) Note that for an experiment, only part of the arguments will be used
    The remaining unused arguments won't affect anything.
    So feel free to register any argument in graphgym.contrib.config
    2) We support *at most* two levels of configs, e.g., cfg.dataset.name

    :return: configuration use by the experiment.
    '''

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #

    # Whether to use GPU
    cfg.use_gpu = False

    # Whether to print verbose logging info
    cfg.verbose = 1

    # Specify the device
    cfg.device = -1

    # Random seed
    cfg.seed = 0

    # Path of configuration file
    cfg.cfg_file = ''

    # For test
    cfg.save_data = False
    cfg.best_res_update_round_wise_key = "val_loss"

    # ------------------------------------------------------------------------ #
    # Early stopping related options
    # ------------------------------------------------------------------------ #
    cfg.early_stop = CN()

    # patience (int): How long to wait after last time the monitored metric improved.
    # Note that the actual_checking_round = patience * cfg.eval.freq
    # To disable the early stop, set the early_stop.patience a integer <=0
    cfg.early_stop.patience = 5
    # delta (float): Minimum change in the monitored metric to indicate an improvement.
    cfg.early_stop.delta = 0.0
    cfg.early_stop.improve_indicator_mode = 'best'  # Early stop when no improve to last `patience` round, in ['mean', 'best']
    cfg.early_stop.the_smaller_the_better = True

    # Monitoring, e.g., 'dissim' for B-local dissimilarity
    cfg.monitoring = []

    # The dir used to save log, exp_config, models, etc,.
    cfg.outdir = 'exp'
    cfg.expname = ''  # detailed exp name to distinguish different sub-exp

    cfg.backend = 'torch'

    # ------------------------------------------------------------------------ #
    # Federate learning related options
    # ------------------------------------------------------------------------ #
    cfg.federate = CN()

    cfg.federate.client_num = 0
    cfg.federate.sample_client_num = -1
    cfg.federate.sample_client_rate = -1.0
    cfg.federate.total_round_num = 50
    cfg.federate.mode = 'standalone'
    cfg.federate.local_update_steps = 1
    cfg.federate.batch_or_epoch = 'batch'
    cfg.federate.share_local_model = False
    cfg.federate.data_weighted_aggr = False  # If True, the weight of aggr is the number of training samples in dataset.
    cfg.federate.online_aggr = False
    cfg.federate.make_global_eval = False
    cfg.federate.standalone = 'local'
    # the method name is used to internally determine composition of different aggregators, messages, handlers, etc.,
    cfg.federate.method = "FedAvg"
    cfg.federate.ignore_weight = False
    cfg.federate.use_ss = False  # Whether to apply Secret Sharing
    cfg.federate.restore_from = ''
    cfg.federate.save_to = ''
    cfg.federate.join_in_info = [
    ]  # The information requirements (from server) for join_in

    # ------------------------------------------------------------------------ #
    # Distribute training related options
    # ------------------------------------------------------------------------ #
    cfg.distribute = CN()

    cfg.distribute.server_host = '0.0.0.0'
    cfg.distribute.server_port = 50050
    cfg.distribute.client_host = '0.0.0.0'
    cfg.distribute.client_port = 50050
    cfg.distribute.role = 'client'
    cfg.distribute.data_file = 'data'
    cfg.distribute.grpc_max_send_message_length = 100 * 1024 * 1024
    cfg.distribute.grpc_max_receive_message_length = 100 * 1024 * 1024
    cfg.distribute.grpc_enable_http_proxy = False

    # ------------------------------------------------------------------------ #
    # Dataset related options
    # ------------------------------------------------------------------------ #
    cfg.data = CN()

    cfg.data.root = 'data'
    cfg.data.type = 'toy'
    cfg.data.batch_size = 64
    cfg.data.drop_last = False
    cfg.data.sizes = [10, 5]
    cfg.data.shuffle = True
    cfg.data.transforms = ''
    cfg.data.pre_transforms = ''
    cfg.data.subsample = 1.0
    cfg.data.splits = [0.8, 0.1, 0.1]  # Train, valid, test splits
    cfg.data.splitter = 'louvain'
    cfg.data.cSBM_phi = [0.5, 0.5, 0.5]
    cfg.data.loader = ''
    cfg.data.num_workers = 0
    cfg.data.graphsaint = CN()
    cfg.data.graphsaint.walk_length = 2
    cfg.data.graphsaint.num_steps = 30

    # ------------------------------------------------------------------------ #
    # Model related options
    # ------------------------------------------------------------------------ #
    cfg.model = CN()

    cfg.model.model_num_per_trainer = 1  # some methods may leverage more than one model in each trainer
    cfg.model.type = 'lr'
    cfg.model.use_bias = True
    cfg.model.task = 'node'
    cfg.model.hidden = 256
    cfg.model.dropout = 0.5
    cfg.model.in_channels = 0  # If 0, model will be built by data.shape
    cfg.model.out_channels = 1
    cfg.model.gnn_layer = 2  # In GPR-GNN, K = gnn_layer
    cfg.model.graph_pooling = 'mean'
    cfg.model.embed_size = 8
    cfg.model.num_item = 0
    cfg.model.num_user = 0

    # ------------------------------------------------------------------------ #
    # Personalization related options
    # ------------------------------------------------------------------------ #
    cfg.personalization = CN()

    # client-distinct param names, e.g., ['pre', 'post']
    cfg.personalization.local_param = []
    cfg.personalization.share_non_trainable_para = False
    cfg.personalization.local_update_steps = -1
    # @regular_weight:
    # The smaller the regular_weight is, the stronger emphasising on personalized model
    # For Ditto, the default value=0.1, the search space is [0.05, 0.1, 0.2, 1, 2]
    # For pFedMe, the default value=15
    cfg.personalization.regular_weight = 0.1

    # @lr:
    # 1) For pFedME, the personalized learning rate to calculate theta approximately using K steps
    # 2) 0.0 indicates use the value according to optimizer.lr in case of users have not specify a valid lr
    cfg.personalization.lr = 0.0

    cfg.personalization.K = 5  # the local approximation steps for pFedMe
    cfg.personalization.beta = 1.0  # the average moving parameter for pFedMe

    # ------------------------------------------------------------------------ #
    # FedSage+ related options
    # ------------------------------------------------------------------------ #
    cfg.fedsageplus = CN()

    cfg.fedsageplus.num_pred = 5
    cfg.fedsageplus.gen_hidden = 128
    cfg.fedsageplus.hide_portion = 0.5
    cfg.fedsageplus.fedgen_epoch = 200
    cfg.fedsageplus.loc_epoch = 1
    cfg.fedsageplus.a = 1.0
    cfg.fedsageplus.b = 1.0
    cfg.fedsageplus.c = 1.0

    # ------------------------------------------------------------------------ #
    # GCFL+ related options
    # ------------------------------------------------------------------------ #
    cfg.gcflplus = CN()

    cfg.gcflplus.EPS_1 = 0.05
    cfg.gcflplus.EPS_2 = 0.1
    cfg.gcflplus.seq_length = 5
    cfg.gcflplus.standardize = False

    # ------------------------------------------------------------------------ #
    # Optimizer related options
    # ------------------------------------------------------------------------ #
    cfg.optimizer = CN()

    cfg.optimizer.type = 'SGD'
    cfg.optimizer.lr = 0.1
    cfg.optimizer.weight_decay = .0
    cfg.optimizer.grad_clip = -1.0  # negative numbers indicate we do not clip grad

    # ------------------------------------------------------------------------ #
    # lr_scheduler related options
    # ------------------------------------------------------------------------ #
    # cfg.lr_scheduler = CN()

    # cfg.lr_scheduler.type = 'StepLR'
    # cfg.lr_scheduler.schlr_params = dict()

    # ------------------------------------------------------------------------ #
    # Criterion related options
    # ------------------------------------------------------------------------ #
    cfg.criterion = CN()

    cfg.criterion.type = 'MSELoss'

    # ------------------------------------------------------------------------ #
    # Trainer related options
    # ------------------------------------------------------------------------ #
    cfg.trainer = CN()

    cfg.trainer.type = 'general'
    cfg.trainer.finetune = CN()
    cfg.trainer.finetune.steps = 0
    cfg.trainer.finetune.only_psn = True
    cfg.trainer.finetune.stepsize = 0.01

    # ------------------------------------------------------------------------ #
    # Evaluation related options
    # ------------------------------------------------------------------------ #
    cfg.eval = CN()

    cfg.eval.freq = 1
    cfg.eval.metrics = []
    cfg.eval.report = ['weighted_avg', 'avg', 'fairness',
                       'raw']  # by default, we report comprehensive results

    # ------------------------------------------------------------------------ #
    # hpo related options
    # ------------------------------------------------------------------------ #
    cfg.hpo = CN()
    cfg.hpo.working_folder = 'hpo'
    cfg.hpo.init_strategy = 'random'
    cfg.hpo.init_cand_num = 16
    cfg.hpo.log_scale = False
    cfg.hpo.larger_better = False
    cfg.hpo.scheduler = 'bruteforce'
    # plot the performanc
    cfg.hpo.plot_interval = 1
    cfg.hpo.metric = 'client_summarized_weighted_avg.test_loss'
    cfg.hpo.sha = CN()
    cfg.hpo.sha.elim_round_num = 3
    cfg.hpo.sha.elim_rate = 3
    cfg.hpo.sha.budgets = []
    cfg.hpo.pbt = CN()
    cfg.hpo.pbt.max_stage = 5
    cfg.hpo.pbt.perf_threshold = 0.1

    # ------------------------------------------------------------------------ #
    # wandb related options
    # ------------------------------------------------------------------------ #
    cfg.wandb = CN()
    cfg.wandb.wandb_use = False
    cfg.wandb.name_user = ''
    cfg.wandb.name_project = ''

    # ------------------------------------------------------------------------ #
    # attack
    # ------------------------------------------------------------------------ #
    cfg.attack = CN()
    cfg.attack.attack_method = ''
    # for gan_attack
    cfg.attack.target_label_ind = -1
    cfg.attack.attacker_id = -1

    # for reconstruct_opt
    cfg.attack.reconstruct_lr = 0.01
    cfg.attack.reconstruct_optim = 'Adam'
    cfg.attack.info_diff_type = 'l2'
    cfg.attack.max_ite = 400
    cfg.attack.alpha_TV = 0.001

    # for active PIA attack
    cfg.attack.alpha_prop_loss = 0

    # for passive PIA attack
    cfg.attack.classifier_PIA = 'randomforest'

    # for gradient Ascent --- MIA attack
    cfg.attack.inject_round = 0

    # ------------------------------------------------------------------------ #
    # Vertical FL related options (for demo)
    # ------------------------------------------------------------------------ #
    cfg.vertical = CN()
    cfg.vertical.use = False
    cfg.vertical.encryption = 'paillier'
    cfg.vertical.dims = [5, 10]
    cfg.vertical.key_size = 3072

    # ------------------------------------------------------------------------ #
    # regularizer related options
    # ------------------------------------------------------------------------ #
    cfg.regularizer = CN()

    cfg.regularizer.type = ''
    cfg.regularizer.mu = 0.

    # ------------------------------------------------------------------------ #
    # nbafl(dp) related options
    # ------------------------------------------------------------------------ #
    cfg.nbafl = CN()

    # Params
    cfg.nbafl.use = False
    cfg.nbafl.mu = 0.
    cfg.nbafl.epsilon = 100.
    cfg.nbafl.w_clip = 1.
    cfg.nbafl.constant = 30.

    # ------------------------------------------------------------------------ #
    # fedopt related options
    # ------------------------------------------------------------------------ #
    cfg.fedopt = CN()

    cfg.fedopt.use = False
    cfg.fedopt.lr_server = 0.01
    cfg.fedopt.type_optimizer = 'SGD'

    # ------------------------------------------------------------------------ #
    # fedprox related options
    # ------------------------------------------------------------------------ #
    cfg.fedprox = CN()

    cfg.fedprox.use = False
    cfg.fedprox.mu = 0.

    # ------------------------------------------------------------------------ #
    # VFL-SGDMF(dp) related options
    # ------------------------------------------------------------------------ #
    cfg.sgdmf = CN()

    cfg.sgdmf.use = False  # if use sgdmf algorithm
    cfg.sgdmf.R = 5.  # The upper bound of rating
    cfg.sgdmf.epsilon = 4.  # \epsilon in dp
    cfg.sgdmf.delta = 0.5  # \delta in dp
    cfg.sgdmf.constant = 1.  # constant
    cfg.sgdmf.theta = -1  # -1 means per-rating privacy, otherwise per-user privacy

    #### Set user customized cfgs
    for func in register.config_dict.values():
        func(cfg)


def assert_cfg(cfg):
    """Checks config values invariants."""
    if cfg.personalization.local_update_steps == -1:
        # By default, use the same step to normal mode
        cfg.personalization.local_update_steps = cfg.federate.local_update_steps

    if cfg.personalization.lr <= 0.0:
        # By default, use the same lr to normal mode
        cfg.personalization.lr = cfg.optimizer.lr

    if cfg.federate.batch_or_epoch not in ['batch', 'epoch']:
        raise ValueError(
            "Value of 'cfg.federate.batch_or_epoch' must be chosen from ['batch', 'epoch']."
        )

    if cfg.backend not in ['torch', 'tensorflow']:
        raise ValueError(
            "Value of 'cfg.backend' must be chosen from ['torch', 'tensorflow']."
        )
    if cfg.backend == 'tensorflow' and cfg.federate.mode == 'standalone':
        raise ValueError(
            "We only support run with distribued mode when backend is tensorflow"
        )
    if cfg.backend == 'tensorflow' and cfg.use_gpu == True:
        raise ValueError(
            "We only support run with cpu when backend is tensorflow")

    # client num related
    assert not (cfg.federate.client_num == 0
                and cfg.federate.mode == 'distributed'
                ), "Please configure the cfg.federate. in distributed mode. "

    # sample client num pre-process
    sample_client_num_valid = (0 < cfg.federate.sample_client_num <=
                               cfg.federate.client_num)
    sample_client_rate_valid = (0 < cfg.federate.sample_client_rate <= 1)
    # (a) sampling case
    if sample_client_rate_valid:
        # (a.1) use sample_client_rate
        cfg.federate.sample_client_num = max(
            1, int(cfg.federate.sample_client_rate * cfg.federate.client_num))
        if sample_client_num_valid:
            logger.warning(
                "Users specify both valid sample_client_rate and sample_client_num, "
                "we will use sample_client_rate")
    # (a.2) use sample_client_num, commented since the below two lines do not change anything
    # elif sample_client_num_valid:
    #     cfg.federate.sample_client_num = cfg.federate.sample_client_num
    if not (sample_client_rate_valid or sample_client_num_valid):
        # (b) non-sampling case, use all clients
        cfg.federate.sample_client_num = cfg.federate.client_num

    if cfg.federate.use_ss:
        assert cfg.federate.client_num == cfg.federate.sample_client_num, \
            "Currently, we support secret sharing only in all-client-participation case"

    if cfg.data.loader == 'graphsaint-rw':
        assert cfg.model.gnn_layer == cfg.data.graphsaint.walk_length, 'Sample size mismatch'
    if cfg.data.loader == 'neighbor':
        assert cfg.model.gnn_layer == len(
            cfg.data.sizes), 'Sample size mismatch'

    # HPO related
    assert cfg.hpo.init_strategy in [
        'full', 'grid', 'random'
    ], "initialization strategy for HPO should be \"full\", \"grid\", or \"random\", but the given choice is {}".format(
        cfg.hpo.init_strategy)
    assert cfg.hpo.scheduler in ['bruteforce', 'sha',
                                 'pbt'], "No HPO scheduler named {}".format(
                                     cfg.hpo.scheduler)
    assert len(cfg.hpo.sha.budgets) == 0 or len(
        cfg.hpo.sha.budgets
    ) == cfg.hpo.sha.elim_round_num, "Either do NOT specify the budgets or specify the budget for each SHA iteration, but the given budgets is {}".format(
        cfg.hpo.sha.budgets)

    # aggregator related
    assert (not cfg.federate.online_aggr) or (
        not cfg.federate.use_ss
    ), "Have not supported to use online aggregator and secrete sharing at the same time"


set_cfg(cfg)
