import os
import sys

import yaml

DEV_MODE = False  # simplify the federatedscope re-setup everytime we change
# the source codes of federatedscope
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.cmd_args import parse_args
from federatedscope.core.configs.config import global_cfg
from federatedscope.autotune import get_scheduler

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

if __name__ == '__main__':
    init_cfg = global_cfg.clone()
    args = parse_args()
    init_cfg.merge_from_file(args.cfg_file)
    init_cfg.merge_from_list(args.opts)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    assert not args.client_cfg_file, 'No support for client-wise config in ' \
                                     'HPO mode.'

    # with open(args.cfg_file, 'r') as ips:
    #     config = yaml.load(ips, Loader=yaml.FullLoader)
    # det_config, tbd_config = split_raw_config(config)
    # global_cfg.merge_from_list(config2cmdargs(det_config))
    # global_cfg.merge_from_list(args.opts)

    scheduler = get_scheduler(init_cfg)
    if init_cfg.hpo.scheduler in ['sha', 'wrap_sha']:
        _ = scheduler.optimize()
    elif init_cfg.hpo.scheduler in [
            'rs', 'bo_kde', 'hb', 'bohb', 'wrap_rs', 'wrap_bo_kde', 'wrap_hb',
            'wrap_bohb'
    ]:
        from federatedscope.autotune.hpbandster import run_hpbandster
        run_hpbandster(init_cfg, scheduler)
    elif init_cfg.hpo.scheduler in [
            'bo_gp', 'bo_rf', 'wrap_bo_gp', 'wrap_bo_rf'
    ]:
        from federatedscope.autotune.smac import run_smac
        run_smac(init_cfg, scheduler)
    else:
        raise ValueError(f'No scheduler named {init_cfg.hpo.scheduler}')

    # logger.info(results)
