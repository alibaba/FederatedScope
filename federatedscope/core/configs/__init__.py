import copy
from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules
    if isfile(f) and not f.endswith('__init__.py')
]

# to ensure the sub-configs registered before set up the global config
all_sub_configs = copy.copy(__all__)
if "config" in all_sub_configs:
    all_sub_configs.remove('config')

from federatedscope.core.configs.config import CN, init_global_cfg
__all__ = __all__ + \
          [
              'CN',
              'init_global_cfg'
          ]

# reorder the config to ensure the base config will be registered first
base_configs = [
    'cfg_data', 'cfg_fl_setting', 'cfg_model', 'cfg_training', 'cfg_evaluation'
]
for base_config in base_configs:
    all_sub_configs.pop(all_sub_configs.index(base_config))
    all_sub_configs.insert(0, base_config)
