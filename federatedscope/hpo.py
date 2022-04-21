import os
import sys
import logging

import yaml

DEV_MODE = True  # simplify the federatedscope re-setup everytime we change the source codes of federatedscope
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

from federatedscope.core.cmd_args import parse_args
from federatedscope.core.configs.config import global_cfg
from federatedscope.autotune import split_raw_config, config2cmdargs, get_scheduler

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

if __name__ == '__main__':
    args = parse_args()
    with open(args.cfg_file, 'r') as ips:
        config = yaml.load(ips, Loader=yaml.FullLoader)
    det_config, tbd_config = split_raw_config(config)
    global_cfg.merge_from_list(config2cmdargs(det_config))
    global_cfg.merge_from_list(args.opts)

    scheduler = get_scheduler(tbd_config)
    results = scheduler.optimize()
    logging.info(results)
