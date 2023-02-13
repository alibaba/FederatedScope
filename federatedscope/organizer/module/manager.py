import abc
import socket
import subprocess
import gradio as gr
import pandas as pd

from contextlib import closing


class Manager(abc.ABC):
    def __init__(self, columns, user, verbose=0):
        self._verbose = verbose
        self._auth = {
            'owner': user,
            'white_list': [],
            'black_list': [],
        }
        self.df = pd.DataFrame(columns=columns)

    def format_logging(self, s, verbose):
        # TODO: implement this
        ...

    def search(self, col, value):
        return self.df[col] == value

    @property
    def auth(self):
        return self._auth

    @auth.setter
    def auth(self, value):
        if not isinstance(value, dict):
            raise gr.Error(f'TypeError auth: {type(value)}')
        for key in self._auth:
            if key in value:
                self._auth[key] = value[key]

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    @abc.abstractmethod
    def display(self):
        raise NotImplementedError

    @abc.abstractmethod
    def add(self):
        raise NotImplementedError

    @abc.abstractmethod
    def shutdown(self):
        raise NotImplementedError

    @staticmethod
    def find_free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    @staticmethod
    def get_cmd_from_pid(pid):
        # TODO: its format might be `[top] <defunct>`, to be fixed
        cmd = subprocess.Popen(f'ps -o cmd fp {pid}'.split(),
                               stdout=subprocess.PIPE).communicate()[0]
        cmd = str(cmd).split("b'CMD\\n")[1]
        if "\\n'" in cmd:
            return cmd[:len("\\n'")]
        else:
            return False

    @staticmethod
    def get_missing_number(numbers):
        begin = 1
        end = len(numbers)
        while begin < end:
            mid = (begin + end) // 2
            if numbers[mid - 1] == mid:
                begin = mid + 1
            else:
                end = mid
        return begin if numbers[begin - 1] != begin else begin + 1
