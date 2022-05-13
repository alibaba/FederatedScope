# Copyright (c) Alibaba, Inc. and its affiliates.
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.fed_runner import FedRunner
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, get_client_cls

def set_config_flitplus(cfg):
    backup_cfg = cfg.clone()

    cfg.use_gpu = False
    cfg.device = 0
    # cfg.early_stop.patience = 20
    # cfg.early_stop.improve_indicator_mode = 'mean'

    cfg.federate.mode = 'standalone'
    # cfg.federate.make_global_eval = True
    cfg.federate.client_num = 5  # [4, 5, 6]
    cfg.federate.total_round_num = 30  # [15, 30, 50]
    cfg.federate.local_update_steps = 10000

    cfg.data.root = 'data/'
    # TODO: Cls: Multi-task ['tox21', 'sider', 'clintox'], Reg: Non convergence, QM9: dataloader
    # cfg.data.type = 'freesolv'  # ['esol', 'freesolv', 'lipo']
    cfg.data.type = 'hiv'  # ['hiv', 'bbbp', 'bace']
    # cfg.data.batch_size = 64
    cfg.data.splitter = 'scaffold_lda'

    cfg.model.type = 'mpnn'
    # cfg.model.type = 'gin'
    cfg.model.hidden = 64
    # cfg.model.out_channels = 1
    cfg.model.out_channels = 2
    cfg.model.task = 'graph'

    cfg.flitplus.alpha = 0.1  # [0.1, 0.5, 1]
    cfg.flitplus.tmpFed = 0.5  # [0.1, 0.5, 1, 2, 5]
    cfg.flitplus.lambdavat = 0.01  # [0.001, 0.01, 0.1, 1]
    cfg.flitplus.factor_ema = 0.8

    cfg.optimizer.type = 'Adam'
    cfg.optimizer.lr = 0.0001
    cfg.optimizer.weight_decay = 0.00001

    # cfg.criterion.type = 'MSELoss'
    cfg.criterion.type = 'CrossEntropyLoss'

    cfg.trainer.type = 'flitplustrainer'

    cfg.eval.freq = 5
    # cfg.eval.metrics = ['rmse']
    cfg.eval.metrics = ['acc', 'correct', 'roc_auc']

    return backup_cfg

if __name__ == '__main__':
    backup_cfg = set_config_flitplus(global_cfg)
    setup_seed(global_cfg.seed)
    update_logger(global_cfg)

    data, modified_cfg = get_data(global_cfg.clone())
    global_cfg.merge_from_other_cfg(modified_cfg)

    Fed_runner = FedRunner(data=data,
                           server_class=get_server_cls(global_cfg),
                           client_class=get_client_cls(global_cfg),
                           config=global_cfg.clone())
    test_best_results = Fed_runner.run()
