import copy
from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules
    if isfile(f) and not f.endswith('__init__.py')
]

# to ensure the sub-configs registered before set up the global config
all_sub_configs_contrib = copy.copy(__all__)
if "config" in all_sub_configs_contrib:
    all_sub_configs_contrib.remove('config')
