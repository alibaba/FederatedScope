import os
import subprocess
import logging
import sys
import time

DEV_MODE = True  # simplify the federatedscope re-setup everytime we change
# the source codes of federatedscope
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

from federatedscope.core.cmd_args import parse_args
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, \
    get_server_cls
from federatedscope.core.configs.config import global_cfg, CfgNode
from federatedscope.core.fed_runner import FedRunner

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    init_cfg.merge_from_list(args.opts)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # load clients' cfg file
    client_cfg = CfgNode.load_cfg(open(args.client_cfg_file,
                                       'r')) if args.client_cfg_file else None

    init_cfg.freeze()

    # Stop the process listening to the server port
    cmd = f"lsof -i tcp:{init_cfg.distribute.server_port}"
    result = subprocess.getoutput(cmd)

    if result != "":
        processes = set()
        for line in result.split("\n"):
            splits = " ".join(line.split()).split(" ")
            pid = splits[1]
            if pid.isdigit():
                processes.add(pid)
        logger.info(f"Killing processes {list(processes)} that are listening to {init_cfg.distribute.server_port}")
        for p in processes:
            os.system(f"kill {p}")

    # Start the main loop
    runner = FedRunner(data=None,
                       server_class=get_server_cls(init_cfg),
                       client_class=get_client_cls(init_cfg),
                       config=init_cfg.clone(),
                       client_config=client_cfg)
    _ = runner.run()
