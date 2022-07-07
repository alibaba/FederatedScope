import copy
import logging
import os

from yacs.config import CfgNode
from yacs.config import _assert_with_logging
from yacs.config import _check_and_coerce_cfg_value_type

import federatedscope.register as register

logger = logging.getLogger(__name__)


class CN(CfgNode):
    """
        An extended configuration system based on [yacs](
        https://github.com/rbgirshick/yacs).
        The two-level tree structure consists of several internal dict-like
        containers to allow simple key-value access and management.

    """
    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        super().__init__(init_dict, key_list, new_allowed)
        self.__dict__["cfg_check_funcs"] = list(
        )  # to check the config values validity

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def register_cfg_check_fun(self, cfg_check_fun):
        self.cfg_check_funcs.append(cfg_check_fun)

    def merge_from_file(self, cfg_filename):
        """
            load configs from a yaml file, another cfg instance or a list
            stores the keys and values.

        :param cfg_filename (string):
        :return:
        """
        super(CN, self).merge_from_file(cfg_filename)
        self.assert_cfg()

    def merge_from_other_cfg(self, cfg_other):
        """
            load configs from another cfg instance

        :param cfg_other (CN):
        :return:
        """
        super(CN, self).merge_from_other_cfg(cfg_other)
        self.assert_cfg()

    def merge_from_list(self, cfg_list):
        """
           load configs from a list stores the keys and values.
           modified `merge_from_list` in `yacs.config.py` to allow adding
           new keys if `is_new_allowed()` returns True

        :param cfg_list (list):
        :return:
        """
        _assert_with_logging(
            len(cfg_list) % 2 == 0,
            "Override list has odd length: {}; it must be a list of pairs".
            format(cfg_list),
        )
        root = self
        for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
            if root.key_is_deprecated(full_key):
                continue
            if root.key_is_renamed(full_key):
                root.raise_key_rename_error(full_key)
            key_list = full_key.split(".")
            d = self
            for subkey in key_list[:-1]:
                _assert_with_logging(subkey in d,
                                     "Non-existent key: {}".format(full_key))
                d = d[subkey]
            subkey = key_list[-1]
            _assert_with_logging(subkey in d or d.is_new_allowed(),
                                 "Non-existent key: {}".format(full_key))
            value = self._decode_cfg_value(v)
            if subkey in d:
                value = _check_and_coerce_cfg_value_type(
                    value, d[subkey], subkey, full_key)
            d[subkey] = value

        self.assert_cfg()

    def assert_cfg(self):
        """
            check the validness of the configuration instance

        :return:
        """
        for check_func in self.cfg_check_funcs:
            check_func(self)

    def clean_unused_sub_cfgs(self):
        """
            Clean the un-used secondary-level CfgNode, whose `.use`
            attribute is `True`

        :return:
        """
        for v in self.values():
            if isinstance(v, CfgNode) or isinstance(v, CN):
                # sub-config
                if hasattr(v, "use") and v.use is False:
                    for k in copy.deepcopy(v).keys():
                        # delete the un-used attributes
                        if k == "use":
                            continue
                        else:
                            del v[k]

    def freeze(self, inform=True):
        """
            1) make the cfg attributes immutable;
            2) save the frozen cfg_check_funcs into
            "self.outdir/config.yaml" for better reproducibility;
            3) if self.wandb.use=True, update the frozen config

        :return:
        """
        self.assert_cfg()
        self.clean_unused_sub_cfgs()
        # save the final cfg
        with open(os.path.join(self.outdir, "config.yaml"), 'w') as outfile:
            from contextlib import redirect_stdout
            with redirect_stdout(outfile):
                tmp_cfg = copy.deepcopy(self)
                tmp_cfg.cfg_check_funcs.clear()
                print(tmp_cfg.dump())
            if self.wandb.use:
                # update the frozen config
                try:
                    import wandb
                except ImportError:
                    logger.error(
                        "cfg.wandb.use=True but not install the wandb package")
                    exit()

                import yaml
                cfg_yaml = yaml.safe_load(tmp_cfg.dump())
                wandb.config.update(cfg_yaml, allow_val_change=True)

        if inform:
            logger.info("the used configs are: \n" + str(tmp_cfg))

        super(CN, self).freeze()


# to ensure the sub-configs registered before set up the global config
from federatedscope.core.configs import all_sub_configs
for sub_config in all_sub_configs:
    __import__("federatedscope.core.configs." + sub_config)

from federatedscope.contrib.configs import all_sub_configs_contrib
for sub_config in all_sub_configs_contrib:
    __import__("federatedscope.contrib.configs." + sub_config)

# Global config object
global_cfg = CN()


def init_global_cfg(cfg):
    r'''
    This function sets the default config value.
    1) Note that for an experiment, only part of the arguments will be used
    The remaining unused arguments won't affect anything.
    So feel free to register any argument in graphgym.contrib.config
    2) We support *at most* two levels of configs, e.g., cfg.dataset.name

    :return: configuration use by the experiment.
    '''

    # ---------------------------------------------------------------------- #
    # Basic options, first level configs
    # ---------------------------------------------------------------------- #

    cfg.backend = 'torch'

    # Whether to use GPU
    cfg.use_gpu = False

    # Whether to print verbose logging info
    cfg.verbose = 1

    # Specify the device
    cfg.device = -1

    # Random seed
    cfg.seed = 0

    # Path of configuration file
    cfg.cfg_file = ''

    # The dir used to save log, exp_config, models, etc,.
    cfg.outdir = 'exp'
    cfg.expname = ''  # detailed exp name to distinguish different sub-exp
    cfg.expname_tag = ''  # detailed exp tag to distinguish different
    # sub-exp with the same expname

    # extend user customized configs
    for func in register.config_dict.values():
        func(cfg)


init_global_cfg(global_cfg)
