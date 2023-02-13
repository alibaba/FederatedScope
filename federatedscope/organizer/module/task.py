from federatedscope.organizer.module.manager import Manager


class TaskManager(Manager):
    def __init__(self, user, organizer):
        super(TaskManager, self).__init__([])
        self.organizer = organizer
        self.user = user

    def display(self):
        raise NotImplementedError

    def add(self):
        raise NotImplementedError

    def shutdown(self):
        raise NotImplementedError
