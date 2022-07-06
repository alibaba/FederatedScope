import collections
import copy
import json
import logging
import math
import os
import random
import signal
import ssl
import time
import urllib.request
from datetime import datetime
from os import path as osp

import numpy as np

# Blind torch
try:
    import torch
    import torchvision
    import torch.distributions as distributions
except ImportError:
    torch = None
    torchvision = None
    distributions = None

logger = logging.getLogger(__name__)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    else:
        import tensorflow as tf
        tf.set_random_seed(seed)


def update_logger(cfg, clear_before_add=False):
    import os
    import logging

    root_logger = logging.getLogger("federatedscope")

    # clear all existing handlers and add the default stream
    if clear_before_add:
        root_logger.handlers = []
        handler = logging.StreamHandler()
        logging_fmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(" \
                      "message)s"
        handler.setFormatter(logging.Formatter(logging_fmt))
        root_logger.addHandler(handler)

    # update level
    if cfg.verbose > 0:
        logging_level = logging.INFO
    else:
        logging_level = logging.WARN
        logger.warning("Skip DEBUG/INFO messages")
    root_logger.setLevel(logging_level)

    # ================ create outdir to save log, exp_config, models, etc,.
    if cfg.outdir == "":
        cfg.outdir = os.path.join(os.getcwd(), "exp")
    if cfg.expname == "":
        cfg.expname = f"{cfg.federate.method}_{cfg.model.type}_on" \
                      f"_{cfg.data.type}_lr{cfg.optimizer.lr}_lste" \
                      f"p{cfg.federate.local_update_steps}"
    cfg.expname = f"{cfg.expname}_{cfg.expname_tag}"
    cfg.outdir = os.path.join(cfg.outdir, cfg.expname)

    # if exist, make directory with given name and time
    if os.path.isdir(cfg.outdir) and os.path.exists(cfg.outdir):
        outdir = os.path.join(cfg.outdir, "sub_exp" +
                              datetime.now().strftime('_%Y%m%d%H%M%S')
                              )  # e.g., sub_exp_20220411030524
        while os.path.exists(outdir):
            time.sleep(1)
            outdir = os.path.join(
                cfg.outdir,
                "sub_exp" + datetime.now().strftime('_%Y%m%d%H%M%S'))
        cfg.outdir = outdir
    # if not, make directory with given name
    os.makedirs(cfg.outdir)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(cfg.outdir, 'exp_print.log'))
    fh.setLevel(logging.DEBUG)
    logger_formatter = logging.Formatter(
        "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    fh.setFormatter(logger_formatter)
    root_logger.addHandler(fh)
    # sys.stderr = sys.stdout

    import socket
    root_logger.info(f"the current machine is at"
                     f" {socket.gethostbyname(socket.gethostname())}")
    root_logger.info(f"the current dir is {os.getcwd()}")
    root_logger.info(f"the output dir is {cfg.outdir}")

    if cfg.wandb.use:
        init_wandb(cfg)


def init_wandb(cfg):
    try:
        import wandb
    except ImportError:
        logger.error("cfg.wandb.use=True but not install the wandb package")
        exit()
    dataset_name = cfg.data.type
    method_name = cfg.federate.method
    exp_name = cfg.expname

    tmp_cfg = copy.deepcopy(cfg)
    tmp_cfg.cfg_check_funcs = []
    import yaml
    cfg_yaml = yaml.safe_load(tmp_cfg.dump())

    wandb.init(project=cfg.wandb.name_project,
               entity=cfg.wandb.name_user,
               config=cfg_yaml,
               group=dataset_name,
               job_type=method_name,
               name=exp_name,
               notes=f"{method_name}, {exp_name}")


def get_dataset(type, root, transform, target_transform, download=True):
    if isinstance(type, str):
        if hasattr(torchvision.datasets, type):
            return getattr(torchvision.datasets,
                           type)(root=root,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
        else:
            raise NotImplementedError('Dataset {} not implement'.format(type))
    else:
        raise TypeError()


def save_local_data(dir_path,
                    train_data=None,
                    train_targets=None,
                    test_data=None,
                    test_targets=None,
                    val_data=None,
                    val_targets=None):
    r"""
    https://github.com/omarfoq/FedEM/blob/main/data/femnist/generate_data.py

    save (`train_data`, `train_targets`) in {dir_path}/train.pt,
    (`val_data`, `val_targets`) in {dir_path}/val.pt
    and (`test_data`, `test_targets`) in {dir_path}/test.pt
    :param dir_path:
    :param train_data:
    :param train_targets:
    :param test_data:
    :param test_targets:
    :param val_data:
    :param val_targets
    """
    if (train_data is not None) and (train_targets is not None):
        torch.save((train_data, train_targets), osp.join(dir_path, "train.pt"))

    if (test_data is not None) and (test_targets is not None):
        torch.save((test_data, test_targets), osp.join(dir_path, "test.pt"))

    if (val_data is not None) and (val_targets is not None):
        torch.save((val_data, val_targets), osp.join(dir_path, "val.pt"))


def filter_by_specified_keywords(param_name, filter_keywords):
    '''
    Arguments:
        param_name (str): parameter name.
    Returns:
        preserve (bool): whether to preserve this parameter.
    '''
    preserve = True
    for kw in filter_keywords:
        if kw in param_name:
            preserve = False
            break
    return preserve


def get_random(type, sample_shape, params, device):
    if not hasattr(distributions, type):
        raise NotImplementedError("Distribution {} is not implemented, "
                                  "please refer to ```torch.distributions```"
                                  "(https://pytorch.org/docs/stable/ "
                                  "distributions.html).".format(type))
    generator = getattr(distributions, type)(**params)
    return generator.sample(sample_shape=sample_shape).to(device)


def batch_iter(data, batch_size=64, shuffled=True):
    assert 'x' in data and 'y' in data
    data_x = data['x']
    data_y = data['y']
    data_size = len(data_y)
    num_batches_per_epoch = math.ceil(data_size / batch_size)

    while True:
        shuffled_index = np.random.permutation(
            np.arange(data_size)) if shuffled else np.arange(data_size)
        for batch in range(num_batches_per_epoch):
            start_index = batch * batch_size
            end_index = min(data_size, (batch + 1) * batch_size)
            sample_index = shuffled_index[start_index:end_index]
            yield {'x': data_x[sample_index], 'y': data_y[sample_index]}


def merge_dict(dict1, dict2):
    # Merge results for history
    for key, value in dict2.items():
        if key not in dict1:
            if isinstance(value, dict):
                dict1[key] = merge_dict({}, value)
            else:
                dict1[key] = [value]
        else:
            if isinstance(value, dict):
                merge_dict(dict1[key], value)
            else:
                dict1[key].append(value)
    return dict1


def download_url(url: str, folder='folder'):
    r"""Downloads the content of an url to a folder.

    Modified from `https://github.com/pyg-team/pytorch_geometric/blob/master
    /torch_geometric/data/download.py`

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        path (string): File path of downloaded files.
    """

    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        logger.info(f'File {file} exists, use existing file.')
        return path

    logger.info(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path


def move_to(obj, device):
    import torch
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")


class Timeout(object):
    def __init__(self, seconds, max_failure=5):
        self.seconds = seconds
        self.max_failure = max_failure

    def __enter__(self):
        def signal_handler(signum, frame):
            raise TimeoutError()

        if self.seconds > 0:
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        signal.alarm(0)

    def reset(self):
        signal.alarm(self.seconds)

    def block(self):
        signal.alarm(0)

    def exceed_max_failure(self, num_failure):
        return num_failure > self.max_failure


def logfile_2_wandb_dict(exp_log_f, raw_out=True):
    """
        parse the logfiles [exp_print.log, eval_results.log] into
        wandb_dict that contains non-nested dicts

    :param exp_log_f: opened exp_log file
    :param raw_out: True indicates "exp_print.log", otherwise indicates
    "eval_results.log",
        the difference is whether contains the logger header such as
        "2022-05-02 16:55:02,843 (client:197) INFO:"

    :return: tuple including (all_log_res, exp_stop_normal, last_line,
    log_res_best)
    """
    log_res_best = {}
    exp_stop_normal = False
    all_log_res = []
    last_line = None
    for line in exp_log_f:
        last_line = line
        exp_stop_normal, log_res = logline_2_wandb_dict(
            exp_stop_normal, line, log_res_best, raw_out)
        if "'Role': 'Server #'" in line:
            all_log_res.append(log_res)
    return all_log_res, exp_stop_normal, last_line, log_res_best


def logline_2_wandb_dict(exp_stop_normal, line, log_res_best, raw_out):
    log_res = {}
    if "INFO:" in line and "Find new best result for" in line:
        # Logger type 1, each line for each metric, e.g.,
        # 2022-03-22 10:48:42,562 (server:459) INFO: Find new best result
        # for client_individual.test_acc with value 0.5911787974683544
        line = line.split("INFO: ")[1]
        parse_res = line.split("with value")
        best_key, best_val = parse_res[-2], parse_res[-1]
        # client_individual.test_acc -> client_individual/test_acc
        best_key = best_key.replace("Find new best result for",
                                    "").replace(".", "/")
        log_res_best[best_key.strip()] = float(best_val.strip())

    if "Find new best result:" in line:
        # each line for all metric of a role, e.g.,
        # Find new best result: {'Client #1': {'val_loss':
        # 132.9812364578247, 'test_total': 36, 'test_avg_loss':
        # 3.709533585442437, 'test_correct': 2.0, 'test_loss':
        # 133.54320907592773, 'test_acc': 0.05555555555555555, 'val_total':
        # 36, 'val_avg_loss': 3.693923234939575, 'val_correct': 4.0,
        # 'val_acc': 0.1111111111111111}}
        line = line.replace("Find new best result: ", "").replace("\'", "\"")
        res = json.loads(s=line)
        for best_type_key, val in res.items():
            for inner_key, inner_val in val.items():
                log_res_best[f"best_{best_type_key}/{inner_key}"] = inner_val

    if "'Role': 'Server #'" in line:
        if raw_out:
            line = line.split("INFO: ")[1]
        res = line.replace("\'", "\"")
        res = json.loads(s=res)
        cur_round = res['Round']
        res.pop('Role')
        if cur_round != "Final" and 'Results_raw' in res:
            res.pop('Results_raw')
        for key, val in res.items():
            if not isinstance(val, dict):
                log_res[key] = val
            else:
                if cur_round != "Final":
                    for key_inner, val_inner in val.items():
                        assert not isinstance(val_inner,
                                              dict), "Un-expected log format"
                        log_res[f"{key}/{key_inner}"] = val_inner

                else:
                    exp_stop_normal = True
                    if key == "Results_raw":
                        for final_type, final_type_dict in res[
                                "Results_raw"].items():
                            for inner_key, inner_val in final_type_dict.items(
                            ):
                                log_res_best[
                                    f"{final_type}/{inner_key}"] = inner_val
    return exp_stop_normal, log_res


def format_log_hooks(hooks_set):
    def format_dict(target_dict):
        print_dict = collections.defaultdict(list)
        for k, v in target_dict.items():
            for element in v:
                print_dict[k].append(element.__name__)
        return print_dict

    if isinstance(hooks_set, list):
        print_obj = [format_dict(_) for _ in hooks_set]
    elif isinstance(hooks_set, dict):
        print_obj = format_dict(hooks_set)
    return json.dumps(print_obj, indent=2).replace('\n', '\n\t')
