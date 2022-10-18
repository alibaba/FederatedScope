from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

__version__ = '0.0.1'


def _setup_logger():
    import logging

    logging_fmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(" \
                  "message)s"
    logger = logging.getLogger("fedhpobench")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(logging_fmt))
    logger.addHandler(handler)
    logger.propagate = False


_setup_logger()
