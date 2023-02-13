import shlex

from datetime import datetime
from collections.abc import MutableMapping

from federatedscope.core.cmd_args import parse_args
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.auxiliaries.data_builder import get_data


class OrganizerLogger:
    def _get_time_stamp(self):
        return f"[{str(datetime.now().strftime('%Y%m%d%H%M%S'))}]"

    def info(self, s):
        print(f"{self._get_time_stamp()} - INFO: {s}")

    def warning(self, s):
        print(f"{self._get_time_stamp()} - WARNING: {s}")

    def error(self, s):
        print(f"{self._get_time_stamp()} - ERROR: {s}")


def anonymize(info, mask):
    for key, value in info.items():
        if key == mask:
            info[key] = "******"
        elif isinstance(value, dict):
            anonymize(info[key], mask)
    return info


def args2yaml(args):
    init_cfg = global_cfg.clone()
    args = parse_args(shlex.split(args))
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    init_cfg.merge_from_list(args.opts)
    _, modified_cfg = get_data(config=init_cfg.clone())
    init_cfg.merge_from_other_cfg(modified_cfg)
    init_cfg.freeze(inform=False, save=False)
    init_cfg.defrost()
    return init_cfg


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for key, value in d.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def config2cmdargs(config):
    """
    Arguments:
        config (dict): key is cfg node name, value is the specified value.
    Returns:
        results (list): cmd args
    """

    results = []
    for key, value in config.items():
        if value and not key.startswith('__'):
            results.append(key)
            results.append(value)
    return results
