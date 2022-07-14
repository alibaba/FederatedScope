# import time
# from server import sendmail
# result = sendmail.delay(dict(to='celery@python.org'))
# while not result.ready():
#     print('Waiting for response... (will re-try in 0.1s)')
#     time.sleep(0.1)
# print(result.get(timeout=1))

from celery import Celery
celery = Celery()
celery.config_from_object('client_config')
celery.send_task('organizer.sendmail', dict(to='celery@python.org'))