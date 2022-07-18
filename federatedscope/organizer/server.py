import os
import time
import redis
import pickle
from celery import Celery


# ---------------------------------------------------------------------- #
# Lobby related (global variable stored in Redis)
# ---------------------------------------------------------------------- #
class Lobby(object):
    def __init__(self, host='localhost', port=6379, db=0):
        self.r = redis.StrictRedis(host=host, port=port, db=db)
        self._set_up(host)

    def _save(self, key, value):
        """
            Save object to Redis via pickle.
        """
        pickled_object = pickle.dumps(value)
        self.r.set(key, pickled_object)

    def _load(self, key):
        """
            Load object from Redis via pickle.
        """
        value = pickle.loads(self.r.get(key))
        return value

    def _set_up(self, host):
        """
           Store all meta info in Redis.
        """
        self._save('localhost', host)
        self._save('blacklist', [])
        # key: room_id, value: configs of FS
        self._save('room', {})

    def _check_room(self):
        """
            Check the validity of the room.
        """
        pass

    def _check_user(self):
        """
            Check the validity of the user (whether in black list, etc).
        """
        pass

    def create_room(self, info, psw=None):
        """
            Create FS server session and store meta info in Redis.
        """
        # Update info in Redis
        room = self._load('room')
        room_id = len(room)
        meta_info = {'info': info, 'psw': psw}
        if room_id in room.keys():
            raise ValueError
        else:
            room[room_id] = meta_info

        # Launch FS
        print(f"python federatedscope/main.py {info}")
        # os.system(f"python federatedscope/main.py {info} &")
        return room_id

    def display_room(self):
        """
            Display all the joinable FS tasks.
        """
        raise NotImplementedError

    def join_room(self, room_id, psw=None):
        """
            Join one specific FS task.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------- #
# Message related
# ---------------------------------------------------------------------- #
organizer = Celery('server',
                   broker='redis://localhost:6379/0',
                   backend='redis://localhost')
organizer.config_from_object('cfg_server')
lobby = Lobby()


@organizer.task
def create_room(info, psw):
    room_id = lobby.create_room(info, psw)
    print(f"The room was created successfully and the id is {room_id}.")
    return room_id


# ---------------------------------------------------------------------- #
# Example
# ---------------------------------------------------------------------- #
@organizer.task
def sendmail(mail):
    print('sending mail to %s...' % mail['to'])
    time.sleep(2.0)
    print('mail sent.')
    return 'mail sent.'
