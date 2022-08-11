import argparse
import sys
from federatedscope.core.configs.config import global_cfg


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='FederatedScope',
                                     add_help=False)
    parser.add_argument('--cfg',
                        dest='cfg_file',
                        help='Config file path',
                        required=False,
                        type=str)
    parser.add_argument('--client_cfg',
                        dest='client_cfg_file',
                        help='Config file path for clients',
                        required=False,
                        default=None,
                        type=str)
    parser.add_argument(
        '--help',
        nargs="?",
        default="all",
    )
    parser.add_argument('opts',
                        help='See federatedscope/core/configs for all options',
                        default=None,
                        nargs=argparse.REMAINDER)
    parse_res = parser.parse_args(args)
    init_cfg = global_cfg.clone()
    # when users type only "main.py" or "main.py help"
    if len(sys.argv) == 1 or parse_res.help == "all":
        parser.print_help()
        init_cfg.print_help()
        sys.exit(1)
    elif hasattr(parse_res, "help") and isinstance(parse_res.help, str):
        init_cfg.print_help(parse_res.help)
        sys.exit(1)
    elif hasattr(parse_res, "help") and isinstance(parse_res.help, list):
        for query in parse_res.help:
            init_cfg.print_help(query)
        sys.exit(1)

    return parse_res
