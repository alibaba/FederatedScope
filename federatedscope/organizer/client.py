import cmd
import time
from celery import Celery

organizer = Celery()
organizer.config_from_object('cfg_client')
""" Celery API examples
command = '--cfg ../../federatedscope/example_configs' \
          '/distributed_femnist_server.yaml'
result = organizer.send_task('server.create_room', [command, 12345])
while not result.ready():
    print('Waiting for response... (will re-try in 1s)')
    time.sleep(1)
print(result.get(timeout=1))

result = organizer.send_task('server.display_room')
while not result.ready():
    print('Waiting for response... (will re-try in 1s)')
    time.sleep(1)
print(result.get(timeout=1))

result = organizer.send_task('server.join_room', [1, 12345])
while not result.ready():
    print('Waiting for response... (will re-try in 1s)')
    time.sleep(1)
print(result.get(timeout=1))
"""


class OrganizerClient(cmd.Cmd):
    intro = 'Welcome to the FS organizer shell. Type help or ? to list ' \
            'commands.\n'
    prompt = 'FederatedScope>> '
    file = None
    ecs = {}
    timeout = 10

    def do_add_ecs(self, ip, user, psw):
        'Add Ecs (ip, user, psw): ADD_ECS 172.X.X.X root 12345'
        self.ecs[ip] = (user, psw)
        print(f"{ip} added.")

    def do_display_ecs(self, arg):
        'Display all saved ECS: DISPLAY_ECS'
        print(self.ecs)

    def do_create_room(self, command, psw=None):
        'Create FS room in server with specific command (command, psw):' \
            'CREATE_ROOM --cfg ../../federatedscope/example_configs' \
            '/distributed_femnist_server.yaml 12345'
        global organizer
        result = organizer.send_task('server.create_room', [command, psw])
        cnt = 0
        while (not result.ready()) and cnt < self.timeout:
            print('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        print(result.get(timeout=1))

    def do_display_room(self, arg):
        'Display all FS rooms: DISPLAY_ROOM'
        global organizer
        result = organizer.send_task('server.display_room')
        cnt = 0
        while (not result.ready()) and cnt < self.timeout:
            print('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        print(result.get(timeout=1))

    def do_join_room(self, room_id, psw=None):
        'Join specific FS room (room_id, psw): JOIN ROOM 0 12345'
        global organizer
        result = organizer.send_task('server.join_room', [room_id, psw])
        cnt = 0
        while (not result.ready()) and cnt < self.timeout:
            print('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        print(result.get(timeout=1))

    def do_quit(self, arg):
        return True


OrganizerClient().cmdloop()
