from federatedscope.core.monitors.monitor import Monitor


class Worker(object):
    """
    The base worker class.
    """
    def __init__(self, ID=-1, state=0, config=None, model=None, strategy=None):
        self._ID = ID
        self._state = state
        self._model = model
        self._cfg = config
        self._strategy = strategy
        self._mode = self._cfg.federate.mode.lower()
        self._monitor = Monitor(config, monitored_object=self)

    @property
    def ID(self):
        return self._ID

    @ID.setter
    def ID(self, value):
        self._ID = value

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, value):
        self._strategy = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value
