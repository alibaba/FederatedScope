# tasks.py
import time
from celery import Celery

organizer = Celery('server', broker='redis://localhost:6379/0', backend='redis://localhost')
organizer.config_from_object('config')

@organizer.task
def sendmail(mail):
    print('sending mail to %s...' % mail['to'])
    time.sleep(2.0)
    print('mail sent.')
    return 'mail sent.'