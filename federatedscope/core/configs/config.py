import copy
import logging
import os

from pathlib import Path

import federatedscope.register as register
from federatedscope.core.configs.yacs_config import CfgNode, _merge_a_into_b, \
    Argument

logger = logging.getLogger(__name__)


def set_help_info(cn_node, help_info_dict, prefix=""):
    for k, v in cn_node.items():
        if isinstance(v, Argument) and k not in help_info_dict:
            help_info_dict[prefix + k] = v.description
        elif isinstance(v, CN):
            set_help_info(v,
                          help_info_dict,
                          prefix=f"{k}." if prefix == "" else f"{prefix}{k}.")


class CN(CfgNode):
    """
    An extended configuration system based on [yacs]( \
    https://github.com/rbgirshick/yacs). \
    The two-level tree structure consists of several internal dict-like \
    containers to allow simple key-value access and management.
    """
    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        init_dict = super().__init__(init_dict, key_list, new_allowed)
        self.__cfg_check_funcs__ = list()  # to check the config values
        # validity
        self.__help_info__ = dict()  # build the help dict

        self.is_ready_for_run = False  # whether this CfgNode has checked its
        # validity, completeness and clean some un-useful info

        if init_dict:
            for k, v in init_dict.items():
                if isinstance(v, Argument):
                    self.__help_info__[k] = v.description
                elif isinstance(v, CN) and "help_info" in v:
                    for name, des in v.__help_info__.items():
                        self.__help_info__[name] = des

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError(name)

    def clear_aux_info(self):
        """
        Clears all the auxiliary information of the CN object.
        """
        if hasattr(self, "__cfg_check_funcs__"):
            delattr(self, "__cfg_check_funcs__")
        if hasattr(self, "__help_info__"):
            delattr(self, "__help_info__")
        if hasattr(self, "is_ready_for_run"):
            delattr(self, "is_ready_for_run")
        for v in self.values():
            if isinstance(v, CN):
                v.clear_aux_info()

    def print_help(self, arg_name=""):
        """
        print help info for a specific given ``arg_name`` or \
        for all arguments if not given ``arg_name``

        Args:
            arg_name: name of specific args
        """
        if arg_name != "" and arg_name in self.__help_info__:
            print(f"  --{arg_name} \t {self.__help_info__[arg_name]}")
        else:
            for k, v in self.__help_info__.items():
                print(f"  --{k} \t {v}")

    def register_cfg_check_fun(self, cfg_check_fun):
        """
        Register a function that checks the configuration node.

        Args:
            cfg_check_fun: function for validation the correctness of cfg.
        """
        self.__cfg_check_funcs__.append(cfg_check_fun)

    def merge_from_file(self, cfg_filename, check_cfg=True):
        """
        load configs from a yaml file, another cfg instance or a list \
        stores the keys and values.

        Args:
            cfg_filename: file name of yaml file
            check_cfg: whether enable ``assert_cfg()``
        """
        cfg_check_funcs = copy.copy(self.__cfg_check_funcs__)
        with open(cfg_filename, "r") as f:
            cfg = self.load_cfg(f)
        self.merge_from_other_cfg(cfg)
        self.__cfg_check_funcs__.clear()
        self.__cfg_check_funcs__.extend(cfg_check_funcs)
        self.assert_cfg(check_cfg)
        set_help_info(self, self.__help_info__)

    def merge_from_other_cfg(self, cfg_other, check_cfg=True):
        """
        load configs from another cfg instance

        Args:
            cfg_other: other cfg to be merged
            check_cfg: whether enable ``assert_cfg()``
        """
        cfg_check_funcs = copy.copy(self.__cfg_check_funcs__)
        _merge_a_into_b(cfg_other, self, self, [])
        self.__cfg_check_funcs__.clear()
        self.__cfg_check_funcs__.extend(cfg_check_funcs)
        self.assert_cfg(check_cfg)
        set_help_info(self, self.__help_info__)

    def merge_from_list(self, cfg_list, check_cfg=True):
        """
        load configs from a list stores the keys and values. \
        modified ``merge_from_list`` in ``yacs.config.py`` to allow adding \
        new keys if ``is_new_allowed()`` returns True \

        Args:
            cfg_list: list of pairs of cfg name and value
            check_cfg: whether enable ``assert_cfg()``
        """
        cfg_check_funcs = copy.copy(self.__cfg_check_funcs__)
        super().merge_from_list(cfg_list)
        self.__cfg_check_funcs__.clear()
        self.__cfg_check_funcs__.extend(cfg_check_funcs)
        self.assert_cfg(check_cfg)
        set_help_info(self, self.__help_info__)

    def assert_cfg(self, check_cfg=True):
        """
        check the validness of the configuration instance

        Args:
            check_cfg: whether enable checks
        """
        if check_cfg:
            for check_func in self.__cfg_check_funcs__:
                check_func(self)

    def clean_unused_sub_cfgs(self):
        """
        Clean the un-used secondary-level CfgNode, whose ``.use`` \
        attribute is ``True``
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

    def check_required_args(self):
        """
        Check required arguments.
        """
        for k, v in self.items():
            if isinstance(v, CN):
                v.check_required_args()
            if isinstance(v, Argument) and v.required and v.value is None:
                logger.warning(f"You have not set the required argument {k}")

    def de_arguments(self):
        """
        some config values are managed via ``Argument`` class, this function \
        is used to make these values clean without the ``Argument`` class, \
        such that the potential type-specific methods work correctly, \
        e.g., ``len(cfg.federate.method)`` for a string config
        """
        for k, v in copy.deepcopy(self).items():
            if isinstance(v, CN):
                self[k].de_arguments()
            if isinstance(v, Argument):
                self[k] = v.value

    def ready_for_run(self, check_cfg=True):
        """
        Check and cleans up the internal state of cfg and save cfg.

        Args:
            check_cfg: whether enable ``assert_cfg()``
        """
        self.assert_cfg(check_cfg)
        self.clean_unused_sub_cfgs()
        self.check_required_args()
        self.de_arguments()
        self.is_ready_for_run = True

    def freeze(self, inform=True, save=True, check_cfg=True):
        """
        (1) make the cfg attributes immutable;
        (2) if ``save==True``, save the frozen cfg_check_funcs into \
            ``self.outdir/config.yaml`` for better reproducibility;
        (3) if ``self.wandb.use==True``, update the frozen config
        """
        self.ready_for_run(check_cfg)
        super(CN, self).freeze()

        if save:  # save the final cfg
            Path(self.outdir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(self.outdir, "config.yaml"),
                      'w') as outfile:
                from contextlib import redirect_stdout
                with redirect_stdout(outfile):
                    tmp_cfg = copy.deepcopy(self)
                    tmp_cfg.clear_aux_info()
                    print(tmp_cfg.dump())
                if self.wandb.use:
                    # update the frozen config
                    try:
                        import wandb
                        import yaml
                        cfg_yaml = yaml.safe_load(tmp_cfg.dump())
                        wandb.config.update(cfg_yaml, allow_val_change=True)
                    except ImportError:
                        logger.error(
                            "cfg.wandb.use=True but not install the wandb "
                            "package")
                        exit()

            if inform:
                logger.info("the used configs are: \n" + str(tmp_cfg))


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
    """
    This function sets the default config value.

    (1) Note that for an experiment, only part of the arguments will be used \
    The remaining unused arguments won't affect anything. \
    So feel free to register any argument in graphgym.contrib.config
    (2) We support more than one levels of configs, e.g., cfg.dataset.name
    """

    # ---------------------------------------------------------------------- #
    # Basic options, first level configs
    # ---------------------------------------------------------------------- #

    cfg.backend = 'torch'

    # Whether to use GPU
    cfg.use_gpu = False

    # Whether to check the completeness of msg_handler
    cfg.check_completeness = False

    # Whether to print verbose logging info
    cfg.verbose = 1

    # How many decimal places we print out using logger
    cfg.print_decimal_digits = 6

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

    set_help_info(cfg, cfg.__help_info__)


init_global_cfg(global_cfg)
