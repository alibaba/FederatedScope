import copy
import json
import logging
import os
import re
import time
import yaml

import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from
    https://stackoverflow.com/a/56944256/3638629"""
    def __init__(self, fmt):
        super().__init__()
        grey = '\x1b[38;21m'
        blue = '\x1b[38;5;39m'
        yellow = "\x1b[33;20m"
        red = '\x1b[38;5;196m'
        bold_red = '\x1b[31;1m'
        reset = '\x1b[0m'

        self.FORMATS = {
            logging.DEBUG: grey + fmt + reset,
            logging.INFO: blue + fmt + reset,
            logging.WARNING: yellow + fmt + reset,
            logging.ERROR: red + fmt + reset,
            logging.CRITICAL: bold_red + fmt + reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class LoggerPrecisionFilter(logging.Filter):
    def __init__(self, precision):
        super().__init__()
        self.print_precision = precision

    def str_round(self, match_res):
        return str(round(eval(match_res.group()), self.print_precision))

    def filter(self, record):
        # use regex to find float numbers and round them to specified precision
        if not isinstance(record.msg, str):
            record.msg = str(record.msg)
        if record.msg != "":
            if re.search(r"([-+]?\d+\.\d+)", record.msg):
                record.msg = re.sub(r"([-+]?\d+\.\d+)", self.str_round,
                                    record.msg)
        return True


def update_logger(cfg, clear_before_add=False, rank=0):
    root_logger = logging.getLogger("federatedscope")

    # clear all existing handlers and add the default stream
    if clear_before_add:
        root_logger.handlers = []
        handler = logging.StreamHandler()
        fmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
        handler.setFormatter(CustomFormatter(fmt))

        root_logger.addHandler(handler)

    # update level
    if rank == 0:
        if cfg.verbose > 0:
            logging_level = logging.INFO
        else:
            logging_level = logging.WARN
            root_logger.warning("Skip DEBUG/INFO messages")
    else:
        root_logger.warning(f"Using deepspeed, and we will disable "
                            f"subprocesses {rank} logger.")
        logging_level = logging.CRITICAL
    root_logger.setLevel(logging_level)

    # ================ create outdir to save log, exp_config, models, etc,.
    if cfg.outdir == "":
        cfg.outdir = os.path.join(os.getcwd(), "exp")
    if cfg.expname == "":
        cfg.expname = f"{cfg.federate.method}_{cfg.model.type}_on" \
                      f"_{cfg.data.type}_lr{cfg.train.optimizer.lr}_lste" \
                      f"p{cfg.train.local_update_steps}"
    if cfg.expname_tag:
        cfg.expname = f"{cfg.expname}_{cfg.expname_tag}"
    cfg.outdir = os.path.join(cfg.outdir, cfg.expname)

    if rank != 0:
        return

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

    # set print precision for terse logging
    np.set_printoptions(precision=cfg.print_decimal_digits)
    precision_filter = LoggerPrecisionFilter(cfg.print_decimal_digits)
    # attach the filter to the fh handler to propagate the filter, since
    # "Filters, unlike levels and handlers, do not propagate",
    # ref https://stackoverflow.com/questions/6850798/why-doesnt-filter-
    # attached-to-the-root-logger-propagate-to-descendant-loggers
    for handler in root_logger.handlers:
        handler.addFilter(precision_filter)

    import socket
    root_logger.info(f"the current machine is at"
                     f" {socket.gethostbyname(socket.gethostname())}")
    root_logger.info(f"the current dir is {os.getcwd()}")
    root_logger.info(f"the output dir is {cfg.outdir}")

    if cfg.wandb.use:
        import sys
        sys.stderr = sys.stdout  # make both stderr and stdout sent to wandb
        # server
        init_wandb(cfg)


def init_wandb(cfg):
    try:
        import wandb
        # on some linux machines, we may need "thread" init to avoid memory
        # leakage
        os.environ["WANDB_START_METHOD"] = "thread"
    except ImportError:
        logger.error("cfg.wandb.use=True but not install the wandb package")
        exit()
    dataset_name = cfg.data.type
    method_name = cfg.federate.method
    exp_name = cfg.expname

    tmp_cfg = copy.deepcopy(cfg)
    if tmp_cfg.is_frozen():
        tmp_cfg.defrost()
    tmp_cfg.clear_aux_info(
    )  # in most cases, no need to save the cfg_check_funcs via wandb
    tmp_cfg.de_arguments()
    cfg_yaml = yaml.safe_load(tmp_cfg.dump())

    wandb.init(project=cfg.wandb.name_project,
               entity=cfg.wandb.name_user,
               config=cfg_yaml,
               group=dataset_name,
               job_type=method_name,
               name=exp_name,
               notes=f"{method_name}, {exp_name}")


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
        # for client_best_individual.test_acc with value 0.5911787974683544
        line = line.split("INFO: ")[1]
        parse_res = line.split("with value")
        best_key, best_val = parse_res[-2], parse_res[-1]
        # client_best_individual.test_acc -> client_best_individual/test_acc
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

    if "'Role'" in line:
        if raw_out:
            line = line.split("INFO: ")[1]
        res = line.replace("\'", "\"")
        res = json.loads(s=res)
        # pre-process the roles
        cur_round = res['Round']
        if "Server" in res['Role']:
            if cur_round != "Final" and 'Results_raw' in res:
                res.pop('Results_raw')
        role = res.pop('Role')
        # parse the k-v pairs
        for key, val in res.items():
            if not isinstance(val, dict):
                log_res[f"{role}, {key}"] = val
            else:
                if cur_round != "Final":
                    if key == "Results_raw":
                        for key_inner, val_inner in res["Results_raw"].items():
                            log_res[f"{role}, {key_inner}"] = val_inner
                    else:
                        for key_inner, val_inner in val.items():
                            assert not isinstance(val_inner, dict), \
                                "Un-expected log format"
                            log_res[f"{role}, {key}/{key_inner}"] = val_inner
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
