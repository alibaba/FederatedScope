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

command = '--cfg federatedscope/example_configs/distributed_femnist_server' \
          '.yaml'
result = organizer.send_task('create_room', [command, 12345])
while not result.ready():
    print('Waiting for response... (will re-try in 1s)')
    time.sleep(1)
print(result.get(timeout=1))
