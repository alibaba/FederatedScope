import redis
import pickle
import subprocess
from celery import Celery

from federatedscope.organizer.utils import anonymize, args2yaml


# ---------------------------------------------------------------------- #
# Lobby related (global variable stored in Redis)
# ---------------------------------------------------------------------- #
class Lobby(object):
    def __init__(self, host='localhost', port=6379, db=0):
        self.r = redis.StrictRedis(host=host, port=port, db=db)
        self.pool = []
        self._set_up()

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
        try:
            value = pickle.loads(self.r.get(key))
        except TypeError:
            value = None
        return value

    def _set_up(self):
        """
           Store all meta info in Redis.
        """
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

    def create_room(self, args, psw=None):
        """
            Create FS server session and store args in Redis.
        """
        self._check_user()
        # Update room args in Redis
        room = self._load('room')
        room_id = str(len(room))
        # TODO: we must convert arg line to yaml dict to avoid conflicts
        #  with port
        meta_info = {
            'command': args,
            'cfg': args2yaml(args),
            'psw': psw,
        }
        if room_id in room.keys():
            raise ValueError
        else:
            room[room_id] = meta_info
        self._save('room', room)

        # Launch FS
        input_args = args.split(' ')
        cmd = ['python', '../../federatedscope/main.py'] + input_args
        p = subprocess.Popen(cmd,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        self.pool.append(p)

        return room_id

    def display_room(self):
        """
            Display all the joinable FS tasks.
        """
        self._check_user()
        mask_key = ['psw', 'cfg']
        room = self._load('room')
        for mask in mask_key:
            room = anonymize(room, mask)
        return room

    def view_room(self, room_id, psw=None):
        """
            View one specific FS task.
        """
        self._check_user()
        room = self._load('room')
        if self._check_room(room, room_id):
            target_room = self._load('room')[room_id]
            if psw != target_room['psw']:
                return 'Wrong Password!'
            else:
                return target_room
        else:
            return 'Target Room is full or invalid, please use ' \
                   '`update_room` to show all available rooms.'

    def shut_down(self):
        """
            Shut down all rooms and kill all subprocesses.
        """
        for p in self.pool:
            p.terminate()
        self._save('room', {})
        return True


# ---------------------------------------------------------------------- #
# Message related
# ---------------------------------------------------------------------- #
organizer = Celery('server',
                   broker='redis://localhost:6379/0',
                   backend='redis://localhost')
organizer.config_from_object('cfg_server')
lobby = Lobby()


# ---------------------------------------------------------------------- #
# Room related tasks
# ---------------------------------------------------------------------- #
@organizer.task
def create_room(args, psw):
    print('Creating room...')
    room_id = lobby.create_room(args, psw)
    rtn_info = f"The room was created successfully and the id is {room_id}."
    print(rtn_info)
    return rtn_info


@organizer.task
def display_room():
    room = lobby.display_room()
    rtn_info = ""
    for key, value in room.items():
        tmp = f"room_id: {key}, info: {value}\n"
        rtn_info += tmp
    print(rtn_info)
    return room


@organizer.task
def view_room(room_id, psw=None):
    rtn_info = lobby.view_room(room_id, psw)
    return rtn_info


@organizer.task
def shut_down():
    lobby.shut_down()
    return 'Shut down all rooms successfully.'
