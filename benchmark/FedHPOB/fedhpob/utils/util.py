import os
import time
import logging
import ConfigSpace as CS

from datetime import datetime


def merge_dict(dict1, dict2):
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


def disable_fs_logger(cfg, clear_before_add=False):
    # Disable FS logger
    root_logger = logging.getLogger("federatedscope")
    # clear all existing handlers and add the default stream
    if clear_before_add:
        root_logger.handlers = []
        handler = logging.StreamHandler()
        logging_fmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(" \
                      "message)s"
        handler.setFormatter(logging.Formatter(logging_fmt))
        root_logger.addHandler(handler)

    root_logger.setLevel(logging.CRITICAL)

    # ================ create outdir to save log, exp_config, models, etc,.
    if cfg.outdir == "":
        cfg.outdir = os.path.join(os.getcwd(), "exp")
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
    fh.setLevel(logging.CRITICAL)
    logger_formatter = logging.Formatter(
        "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    fh.setFormatter(logger_formatter)
    root_logger.addHandler(fh)


def cfg2name(cfg):
    repeat = 0
    dir = os.path.join(
        cfg.benchmark.out_dir,
        f'{cfg.benchmark.data}_{cfg.benchmark.model}_{cfg.benchmark.type}'
        f'_{cfg.benchmark.algo}')
    os.makedirs(dir, exist_ok=True)
    while os.path.exists(
            os.path.join(dir, f'{cfg.optimizer.type}_repeat{repeat}.txt')):
        repeat += 1
    return os.path.join(dir, f'{cfg.optimizer.type}_repeat{repeat}.txt')


def dict2cfg(space):
    configuration_space = CS.ConfigurationSpace()
    for key, value in space.items():
        hyperparameter = CS.CategoricalHyperparameter(key, choices=value)
        configuration_space.add_hyperparameter(hyperparameter)
    return configuration_space
