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
        self.save('info', '')

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

    def create_room(self, info, psw):
        """
            Create FS server session and store meta info in Redis.
        """
        raise NotImplementedError

    def display_room(self):
        """
            Display all joinable FS tasks.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------- #
# Message related
# ---------------------------------------------------------------------- #
organizer = Celery('server',
                   broker='redis://localhost:6379/0',
                   backend='redis://localhost')
organizer.config_from_object('cfg_server')


# ---------------------------------------------------------------------- #
# Example
# ---------------------------------------------------------------------- #
@organizer.task
def sendmail(mail):
    print('sending mail to %s...' % mail['to'])
    time.sleep(2.0)
    print('mail sent.')
    return 'mail sent.'
