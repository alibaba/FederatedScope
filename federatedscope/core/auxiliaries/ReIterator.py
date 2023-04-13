class ReIterator:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)
        self.reset_flag = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            item = next(self.iterator)
        except StopIteration:
            self.reset()
            item = next(self.iterator)
        return item

    def reset(self):
        self.iterator = iter(self.loader)
