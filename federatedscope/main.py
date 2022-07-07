import os
import sys

from yacs.config import CfgNode

DEV_MODE = False  # simplify the federatedscope re-setup everytime we change
# the source codes of federatedscope
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

from federatedscope.core.cmd_args import parse_args
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, update_logger
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, \
    get_server_cls
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.fed_runner import FedRunner

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

if __name__ == '__main__':
    init_cfg = global_cfg.clone()
    args = parse_args()
    init_cfg.merge_from_file(args.cfg_file)
    init_cfg.merge_from_list(args.opts)

    update_logger(init_cfg)
    setup_seed(init_cfg.seed)

    # load clients' cfg file
    client_cfg = CfgNode.load_cfg(open(args.client_cfg_file,
                                       'r')) if args.client_cfg_file else None

    # federated dataset might change the number of clients
    # thus, we allow the creation procedure of dataset to modify the global
    # cfg object
    data, modified_cfg = get_data(config=init_cfg.clone())
    init_cfg.merge_from_other_cfg(modified_cfg)

    init_cfg.freeze()

    runner = FedRunner(data=data,
                       server_class=get_server_cls(init_cfg),
                       client_class=get_client_cls(init_cfg),
                       config=init_cfg.clone(),
                       client_config=client_cfg)
    _ = runner.run()
