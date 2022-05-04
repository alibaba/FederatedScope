import copy
from os.path import dirname, basename, isfile, join
import glob
from federatedscope.core.configs.config import CN

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules
    if isfile(f) and not f.endswith('__init__.py')
]

__all__ = __all__ + ['CN']

# to ensure the sub-configs registered before set up the global config
all_sub_configs = copy.copy(__all__)
if "config" in all_sub_configs:
    all_sub_configs.remove('config')
