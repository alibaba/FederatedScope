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
    cfg.federate.make_global_eval = True
    cfg.federate.client_num = 5
    cfg.federate.total_round_num = 400
    cfg.federate.local_update_steps = 16

    cfg.data.root = 'data/'
    # Cls: ['bbbp', 'bace', 'tox21', 'sider', 'clintox'], Reg: ['esol', 'freesolv', 'lipo']
    cfg.data.type = 'esol'
    cfg.data.splitter = 'scaffold_lda'

    cfg.model.type = 'sage'
    cfg.model.hidden = 64
    cfg.model.out_channels = 1
    cfg.model.task = 'graph'

    cfg.flitplus.alpha = 0.1
    cfg.flitplus.tmpFed = 0.5
    cfg.flitplus.lambdavat = 0.01

    cfg.optimizer.lr = 0.25
    cfg.optimizer.weight_decay = 0.0005

    cfg.criterion.type = 'CrossEntropyLoss'
    cfg.trainer.type = 'flitplustrainer'
    cfg.eval.freq = 5
    cfg.eval.metrics = ['acc', 'correct']

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
