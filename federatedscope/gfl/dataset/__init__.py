#from __future__ import absolute_import
#from __future__ import print_function
#from __future__ import division
#
#from federatedscope.gfl.dataset.acm import ACM
#from federatedscope.gfl.dataset.recsys import RecSys
#from federatedscope.gfl.dataset.dblp import DBLP
#from federatedscope.gfl.dataset.dblpfull import DBLPfull
#from federatedscope.gfl.dataset.dblp_new import DBLPNew
#
#
#__all__ = ['ACM', 'RecSys', 'DBLP', 'DBLPfull', 'DBLPNew']
#
from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules
    if isfile(f) and not f.endswith('__init__.py')
]
