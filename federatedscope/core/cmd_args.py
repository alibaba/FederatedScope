import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='FederatedScope')
    parser.add_argument('--cfg',
                        dest='cfg_file',
                        help='Config file path',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help='See federatedscope/core/configs for all options',
                        default=None,
                        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # To support the format "para_a=val_a"
    args = parser.parse_args()
    res = []
    for item in args.opts:
        if "=" in item:
            para, val = item.split("=")
            res.append(para)
            res.append(val)
        else:
            res.append(item)
    args.opts = res

    return args
