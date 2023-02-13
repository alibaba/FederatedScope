from federatedscope.organizer.module.manager import Manager


class ECSManager(Manager):
    def __init__(self):
        super(ECSManager, self).__init__([])

    def display(self):
        raise NotImplementedError

    def add(self):
        raise NotImplementedError

    def shutdown(self):
        raise NotImplementedError
