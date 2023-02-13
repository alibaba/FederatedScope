# TODO: delete this file as we use hpo.use in main

import os
import sys

DEV_MODE = False  # simplify the federatedscope re-setup everytime we change
# the source codes of federatedscope
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.configs.config import global_cfg, CfgNode
from federatedscope.autotune import get_scheduler, run_scheduler

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

if __name__ == '__main__':
    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    # Update Exp_name for hpo
    if init_cfg.expname == '':
        from federatedscope.autotune.utils import generate_hpo_exp_name
        init_cfg.expname = generate_hpo_exp_name(init_cfg)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # load clients' cfg file
    if args.client_cfg_file:
        client_cfgs = CfgNode.load_cfg(open(args.client_cfg_file, 'r'))
        # client_cfgs.set_new_allowed(True)
        client_cfgs.merge_from_list(client_cfg_opt)
    else:
        client_cfgs = None

    scheduler = get_scheduler(init_cfg, client_cfgs)
    run_scheduler(scheduler, init_cfg, client_cfgs)
