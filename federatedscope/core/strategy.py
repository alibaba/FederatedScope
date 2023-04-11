import sys


class Strategy(object):
    def __init__(self, stg_type=None, threshold=0):
        self._stg_type = stg_type
        self._threshold = threshold

    @property
    def stg_type(self):
        return self._stg_type

    @stg_type.setter
    def stg_type(self, value):
        self._stg_type = value

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value
