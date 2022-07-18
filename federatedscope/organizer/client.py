import cmd
import time
from celery import Celery

organizer = Celery()
organizer.config_from_object('cfg_client')


class OrganizerClient(cmd.Cmd):
    intro = 'Welcome to the FS organizer shell. Type help or ? to list ' \
            'commands.\n'
    prompt = 'FederatedScope>> '
    ecs = {}
    timeout = 10

    def do_add_ecs(self, line):
        'Add Ecs (ip, user, psw): add_ecs 172.X.X.X root 12345 port'
        ip, user, psw, port = line.split(' ')
        self.ecs[len(self.ecs)] = {
            'ip': ip,
            'user': user,
            'psw': psw,
            'port': port
        }
        print(f"{ip} added.")
        # TODO: set up FS here?

    def do_display_ecs(self, line):
        'Display all saved ECS: display_ecs'
        info = ""
        for key, value in self.ecs.items():
            info += f"ecs_id: {key}, info: {value}\n"
        print(info)

    def do_create_room(self, line):
        'Create FS room in server with specific command (command, psw):' \
            'create_room --cfg ../../federatedscope/example_configs' \
            '/distributed_femnist_server.yaml 12345'
        global organizer
        psw = line.split(' ')[-1]
        command = line[:-len(psw) - 1]
        result = organizer.send_task('server.create_room', [command, psw])
        cnt = 0
        while (not result.ready()) and cnt < self.timeout:
            print('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        print(result.get(timeout=1))

    def do_display_room(self, line):
        'Display all FS rooms: display_room'
        global organizer
        result = organizer.send_task('server.display_room')
        cnt = 0
        while (not result.ready()) and cnt < self.timeout:
            print('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        print(result.get(timeout=1))

    def do_join_room(self, line):
        'Join specific FS room (room_id, psw): JOIN ROOM 0 12345'
        global organizer
        room_id, psw = line.split(' ')
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
