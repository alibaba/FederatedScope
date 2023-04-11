import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='FederatedScope')
    parser.add_argument('--cfg',
                        dest='cfg_file',
                        help='Config file path',
                        required=True,
                        type=str)
    parser.add_argument('--client_cfg',
                        dest='client_cfg_file',
                        help='Config file path for clients',
                        required=False,
                        default=None,
                        type=str)
    parser.add_argument('opts',
                        help='See federatedscope/core/configs for all options',
                        default=None,
                        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()
