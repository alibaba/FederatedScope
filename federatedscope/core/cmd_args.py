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
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument(
        '--help',
        nargs="?",
        const="all",
        default="",
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
    elif hasattr(parse_res, "help") and isinstance(
            parse_res.help, str) and parse_res.help != "":
        init_cfg.print_help(parse_res.help)
        sys.exit(1)
    elif hasattr(parse_res, "help") and isinstance(
            parse_res.help, list) and len(parse_res.help) != 0:
        for query in parse_res.help:
            init_cfg.print_help(query)
        sys.exit(1)

    return parse_res


def parse_client_cfg(arg_opts):
    """
    Arguments:
        arg_opts: list pairs of arg.opts
    """
    client_cfg_opts = []
    i = 0
    while i < len(arg_opts):
        if arg_opts[i].startswith('client'):
            client_cfg_opts.append(arg_opts.pop(i))
            client_cfg_opts.append(arg_opts.pop(i))
        else:
            i += 1
    return arg_opts, client_cfg_opts
