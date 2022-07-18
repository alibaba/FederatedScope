import time
from celery import Celery

organizer = Celery()
organizer.config_from_object('cfg_client')
# result = sendmail.delay(dict(to='celery@python.org'))
result = organizer.send_task('server.sendmail', [dict(to='celery@python.org')])
while not result.ready():
    print('Waiting for response... (will re-try in 1s)')
    time.sleep(1)
print(result.get(timeout=1))
