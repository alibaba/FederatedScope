import subprocess
import redis
import pickle
from celery import Celery

from federatedscope.organizer.utils import anonymize


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

    def _check_room(self, room, room_id):
        """
            Check the validity of the room.
        """
        if room_id in room.keys():
            return True
        else:
            # TODO: check whether the room is full
            return False

    def _check_user(self):
        """
            Check the validity of the user (whether in black list, etc).
        """
        pass

    def create_room(self, info, psw=None):
        """
            Create FS server session and store meta info in Redis.
        """
        self._check_user()
        # Update room info in Redis
        room = self._load('room')
        room_id = len(room)
        meta_info = {'info': info, 'psw': psw}
        if room_id in room.keys():
            raise ValueError
        else:
            room[room_id] = meta_info
        self._save('room', room)
        # Launch FS
        info = info.split(' ')
        cmd = ['python', '../../federatedscope/main.py'] + info
        subprocess.Popen(cmd,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        return room_id

    def display_room(self):
        """
            Display all the joinable FS tasks.
        """
        self._check_user()
        room = anonymize(self._load('room'), 'psw')
        return room

    def join_room(self, room_id, psw=None):
        """
            Join one specific FS task.
        """
        self._check_user()
        room = self._load('room')
        if self._check_room(room, room_id):
            target_room = self._load('room')[room_id]
            if psw != target_room['psw']:
                return 'Wrong Password!'
            else:
                return target_room['info']
        else:
            return 'Target Room is full or invalid, please use ' \
                   '`display_room` to show all available rooms.'


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
    print('Creating room...')
    room_id = lobby.create_room(info, psw)
    rtn_info = f"The room was created successfully and the id is {room_id}."
    print(rtn_info)
    return rtn_info


@organizer.task
def display_room():
    room = lobby.display_room()
    rtn_info = f"Room: {room}"
    return rtn_info


@organizer.task
def join_room(room_id, psw=None):
    rtn_info = lobby.join_room(room_id, psw)
    return rtn_info
