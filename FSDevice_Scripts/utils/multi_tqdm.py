import collections
import time
from multiprocessing import Queue, Process


def handle_msg(queue: Queue):
    while True:
        obj = queue.get()
        msg_type, args = obj
        if msg_type.lower() == "update":
            name, value = args
            MultiProcessBar.update(name, value)
        elif msg_type.lower() == "close":
            MultiProcessBar.close()
        elif msg_type.lower() == "init":
            names = args
            MultiProcessBar.init(names)
        elif msg_type.lower() == "end":
            break
        else:
            raise RuntimeError()


class MultiProcessBar(object):
    pros = collections.defaultdict(int)
    LENGTH = 85

    @classmethod
    def update(cls, name, value):
        cls.pros[name] = int(value)
        cls.plot()

    @classmethod
    def close(cls):
        for name in cls.pros.keys():
            cls.pros[name] = 0

    @classmethod
    def set_names(cls, names):
        for name in names:
            cls.pros[name] = 0

    @classmethod
    def init(cls):
        for name in cls.pros.keys():
            cls.pros[name] = 0
        cls.plot(back=False)

    @classmethod
    def get_prefix_len(cls):
        return max([len(_) for _ in cls.pros.keys()] + [0])

    @classmethod
    def plot(cls, back=True):
        prefix_len = cls.get_prefix_len()
        bars = list()
        bar_len = cls.LENGTH - prefix_len - 21
        for name, value in cls.pros.items():
            length = int(value / 100. * bar_len)
            bars.append(
                f"MACHINE {name:<{prefix_len}} " + f":[{''.center(length, '#') + ''.center(bar_len - length, '.')}] {value:>3}/100%")
        if back:
            print("\033[F" * len(bars) + "\n".join(bars))
        else:
            print("\n".join(bars))
