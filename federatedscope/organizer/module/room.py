import time

import pandas as pd

from federatedscope.core.configs.config import global_cfg
from federatedscope.organizer.utils import config2cmdargs, flatten_dict
from federatedscope.organizer.module.manager import Manager
from federatedscope.organizer.cfg_client import TIMEOUT


class RoomManager(Manager):
    def __init__(self, user, organizer, logger):
        super(RoomManager, self).__init__([
            'idx', 'abstract', 'cfg', 'auth', 'log_file', 'port', 'process',
            'cur_client', 'max_client'
        ],
                                          user=user)
        self.organizer = organizer
        self.logger = logger

    def display(self, condition={}):
        # Sync and display lobby
        result = self.organizer.send_task('server.display_lobby', [self.auth])

        cnt = 0
        while (not result.ready()) and cnt < TIMEOUT:
            self.logger.info('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        # TODO: merge remote df with local df
        msg = result.get(timeout=1)
        if isinstance(msg, str):
            self.logger.info(msg)
        elif isinstance(msg, pd.DataFrame):
            lobby = msg
            if len(lobby) == 0:
                self.logger.info(
                    'No task available now. Please create a new task.')
            else:
                for i in range(len(lobby)):
                    room = lobby.loc[i]

        else:
            self.logger.info(f'TypeError {type(msg)}')

        # Filter by condition
        if condition:
            # convert `str` to `dict`
            for key, value in condition.items():
                lobby = lobby.loc[lobby[key] == value]

        self.logger.info(self.tmp_lobby)

    def add(self, room_type, yaml, opts, password='123'):
        opts = opts.split(' ')
        cfg = global_cfg.clone()
        if yaml.endswith('.yaml'):
            cfg.merge_from_file(yaml)
        else:
            self.logger.warning('The yaml file is none or invalid, ignored.')
        cfg.merge_from_list(opts)
        cfg = config2cmdargs(flatten_dict(cfg))

        command = ''
        for i in cfg:
            value = f'{i}'.replace(' ', '')
            command += f' "{value}"'
        args = command[1:]
        result = self.organizer.send_task(
            'server.add_room', [room_type, args, password, self.auth])
        cnt = 0
        while (not result.ready()) and cnt < TIMEOUT:
            self.logger.info('Waiting for response... (will re-try in 1s)')
            TIMEOUT.sleep(1)
            cnt += 1
        self.logger.info(result.get(timeout=1))
        return True

    def auth(self, idx, password):
        # Get permission for unauthorized room
        # Send idx, password, user to verify
        raise NotImplementedError

    def shutdown(self, idx):
        # Shut down room
        raise NotImplementedError


if __name__ == '__main__':
    import logging
    from celery import Celery

    organizer = Celery()
    logger = logging.getLogger(__name__)

    rm = RoomManager('root', organizer, logger)

    # Test functions
    rm.display()
    rm.add()
    rm.auth()
    rm.shutdown()
