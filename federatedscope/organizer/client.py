import time
from celery import Celery

organizer = Celery()
organizer.config_from_object('cfg_client')

command = '--cfg federatedscope/example_configs/distributed_femnist_server' \
          '.yaml'
result = organizer.send_task('server.create_room', [command, 12345])
while not result.ready():
    print('Waiting for response... (will re-try in 1s)')
    time.sleep(1)
print(result.get(timeout=1))
