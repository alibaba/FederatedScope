import copy
import os

from yacs.config import CfgNode

import federatedscope.register as register


class CN(CfgNode):
    """
        An extended configuration system based on [yacs](https://github.com/rbgirshick/yacs).
        The two-level tree structure consists of several internal dict-like containers to allow simple key-value access and management.

    """
    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        super().__init__(init_dict, key_list, new_allowed)
        self.cfg_check_funcs = []  # to check the config values validity

    def register_cfg_check_fun(self, cfg_check_fun):
        self.cfg_check_funcs.append(cfg_check_fun)

    def merge_from_file(self, cfg_filename):
        """
            load configs from a yaml file, another cfg instance or a list stores the keys and values.

        :param cfg_filename (string):
        :return:
        """
        cfg_check_funcs = copy.copy(self.cfg_check_funcs)
        super(CN, self).merge_from_file(cfg_filename)
        self.cfg_check_funcs = cfg_check_funcs
        self.assert_cfg()

    def merge_from_other_cfg(self, cfg_other):
        """
            load configs from another cfg instance

        :param cfg_other (CN):
        :return:
        """
        cfg_check_funcs = copy.copy(self.cfg_check_funcs)
        super(CN, self).merge_from_other_cfg(cfg_other)
        self.cfg_check_funcs = cfg_check_funcs
        self.assert_cfg()

    def merge_from_list(self, cfg_list):
        """
           load configs from a list stores the keys and values.

        :param cfg_list (list):
        :return:
        """
        cfg_check_funcs = copy.copy(self.cfg_check_funcs)
        super(CN, self).merge_from_list(cfg_list)
        self.cfg_check_funcs = cfg_check_funcs
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
            Clean the un-used secondary-level CfgNode, whose `.use` attribute is `True`

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

    def freeze(self):
        """
            make the cfg attributes immutable, and save the freezed cfg_check_funcs into "self.outdir/config.yaml" for better reproducibility

        :return:
        """
        self.assert_cfg()
        self.clean_unused_sub_cfgs()
        # save the final cfg
        with open(os.path.join(self.outdir, "config.yaml"), 'w') as outfile:
            from contextlib import redirect_stdout
            with redirect_stdout(outfile):
                tmp_cfg = copy.deepcopy(self)
                tmp_cfg.cfg_check_funcs = []
                print(tmp_cfg.dump())

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

    # ------------------------------------------------------------------------ #
    # Basic options, first level configs
    # ------------------------------------------------------------------------ #

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

    # extend user customized configs
    for func in register.config_dict.values():
        func(cfg)


init_global_cfg(global_cfg)
