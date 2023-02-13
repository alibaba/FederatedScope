import redis
import pickle
import subprocess
from datetime import datetime

from federatedscope.organizer.module.manager import Manager
from federatedscope.organizer.utils import args2yaml, config2cmdargs, \
    flatten_dict


class Lobby(Manager):
    def __init__(self, host='localhost', port=6379, db=0):
        super(Lobby, self).__init__([
            'idx', 'abstract', 'cfg', 'password', 'auth', 'log_file', 'port',
            'process', 'cur_client', 'max_client'
        ],
                                    user='root')
        self.database = redis.StrictRedis(host=host, port=port, db=db)
        self._save('lobby', self.df)
        self._save('auth', self._auth)

    def _save(self, key, value):
        """
        Save object to Redis via pickle.
        """
        pickled_object = pickle.dumps(value)
        self.database.set(key, pickled_object)

    def _load(self, key):
        """
        Load object from Redis via pickle.
        """
        try:
            value = pickle.loads(self.database.get(key))
        except TypeError:
            value = None
        return value

    def _refresh_lobby(self):
        """
        Refresh room status and remove finished or dead room.
        """
        dead_pids = []
        lobby = self._load('lobby')
        for i in range(len(lobby)):
            pid = lobby.loc[i]['process'].pid
            if not self.get_cmd_from_pid(pid):
                dead_pids.append(i)
        if dead_pids:
            lobby.drop(dead_pids)
            self._save('lobby', lobby)

    def _check_user(self, user, is_root=False):
        """
        Check the validity of the user. If white list is enabled, user must
        be in the white list. If white list is not enabled, user must not be in
        the black list.
        """
        auth = self._load('auth')

        if is_root:
            return user == auth['owner']

        if len(auth['white_list']) > 0:
            if user not in auth['white_list']:
                return False
        else:
            if user in auth['black_list']:
                return False
        return True

    def add(self, args, password, auth):
        """
        Create FS server session and store args in Redis.
        """
        if not self._check_user(auth['owner']):
            return 'You are not permitted！'
        self._refresh_lobby()
        lobby = self._load('lobby')
        # Update room args in Redis
        new_room_idx = self.get_missing_number(list(lobby['idx']))
        cfg = args2yaml(args)

        room = {
            'idx': new_room_idx,
            'abstract': f'{cfg.data.type} {cfg.model.type}',  # TODO: prettify
            'cfg': config2cmdargs(flatten_dict(cfg)),
            'password': password,
            'auth': auth,
            'log_file': str(datetime.now().strftime('_%Y%m%d%H%M%S')) + '.out',
            'port': self.find_free_port,
            'process': None,  # default, to be updated after launch
            'cur_client': 0,
            'max_client': cfg.federate.client_num
        }

        # Launch FS
        input_args = args.split(' ')
        cmd = ['nohup', 'python', '../../federatedscope/main.py'] + \
            input_args + ['>', room['log_file']]
        p = subprocess.Popen(cmd,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        # Update process
        room['process'] = p

        # Update lobby
        lobby.loc[len(lobby)] = room
        self._save('lobby', lobby)
        return f"The room was created successfully with {room}."

    def display(self, auth):
        """
        Display FS lobby.
        """
        if not self._check_user(auth['owner']):
            return 'You are not permitted！'

        self._refresh_lobby()
        mask_key = ['cfg', 'password']  # Important information
        lobby = self._load('lobby')
        for mask in mask_key:
            lobby.drop(mask)
        return lobby

    def auth(self, idx, password, auth):
        """
        Auth and send key of certain room back.
        """
        if not self._check_user(auth['owner']):
            return 'You are not permitted！'

        self._refresh_lobby()
        lobby = self._load('lobby')
        if int(idx) in list(lobby['idx']):
            # Check the validity of the room
            room = lobby.loc[lobby['idx'] == int(idx)].loc[0]
            if room['cur_client'] < room['max_client']:
                # Joinable, check auth and password
                room_auth, user = room['auth'], auth['owner']
                if len(room_auth['white_list']) > 0:
                    if user not in room_auth['white_list']:
                        return f'You are not in the white list of room {idx}'
                else:
                    if user in room_auth['black_list']:
                        return f'You are in the black list of room {idx}'

                # Check password
                if password != room['password']:
                    return 'Wrong Password!'
                else:
                    return room
            else:
                # Full
                return f'Room {idx} is full'
        else:
            # Room does not exist
            return f'Room {idx} does not exist'

    def shutdown(self, idx, auth):
        """
        Shut down all or a certain room
        """
        if idx:
            if not self._check_user(auth['owner']):
                return 'You are not permitted！'
            self._refresh_lobby()
            lobby = self._load('lobby')

            room = lobby.loc[lobby['idx'] == int(idx)].loc[0]
            room_auth, user = room['auth'], auth['owner']
            if room_auth['owner'] == user:
                room['process'].terminate()
                return f'Room {idx} shut down successfully.'
            else:
                return 'You are not permitted'
        else:
            if not self._check_user(auth['owner'], is_root=True):
                return 'You are not permitted！'
            else:
                self._refresh_lobby()
                lobby = self._load('lobby')
                for p in lobby['process']:
                    p.terminate()
                return 'Shut down all rooms successfully.'
