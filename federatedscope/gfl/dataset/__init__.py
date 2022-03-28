#from __future__ import absolute_import
#from __future__ import print_function
#from __future__ import division
#
#from flpackage.gfl.dataset.acm import ACM
#from flpackage.gfl.dataset.recsys import RecSys
#from flpackage.gfl.dataset.dblp import DBLP
#from flpackage.gfl.dataset.dblpfull import DBLPfull
#from flpackage.gfl.dataset.dblp_new import DBLPNew
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
