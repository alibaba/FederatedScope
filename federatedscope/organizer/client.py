import cmd
import time
from celery import Celery

from federatedscope.organizer.utils import SSHManager

organizer = Celery()
organizer.config_from_object('cfg_client')


class OrganizerClient(cmd.Cmd):
    intro = 'Welcome to the FS organizer shell. Type help or ? to list ' \
            'commands.\n'
    prompt = 'FederatedScope>> '
    ecs = {}
    room = {}
    timeout = 10

    # ---------------------------------------------------------------------- #
    # SSH Manager related
    # ---------------------------------------------------------------------- #
    def do_add_ecs(self, line):
        'Add Ecs (ip, user, psw, port): add_ecs 172.X.X.X root 12345 50002'
        try:
            ip, user, psw, port = line.split(' ')
            key = f"{ip}:{port}"
            if key in self.ecs:
                raise ValueError(f"ECS `{key}` already exists.")
            self.ecs[key] = SSHManager(ip, user, psw, port)
            print(f"{self.ecs[key]} added.")
        except Exception as error:
            print(f"Exception: {error}")

    def do_del_ecs(self, line):
        'Delete Ecs (ip, port): del_ecs 172.X.X.X 50002'
        try:
            ip, port = line.split(' ')
            key = f"{ip}:{port}"
            print(f"Delete {key} {self.ecs[key]}.")
            del self.ecs[key]
        except Exception as error:
            print(f"Exception: {error}")

    def do_display_ecs(self, line):
        'Display all saved ECS: display_ecs'
        try:
            info = ""
            for key, value in self.ecs.items():
                info += f"ecs: {key}, info: {value}\n"
            print(info)
        except Exception as error:
            print(f"Exception: {error}")

    def do_join_room(self, line):
        'Let a ECS join specific FS: display_ecs'
        # TODO: join room
        pass

    # ---------------------------------------------------------------------- #
    # Message related
    # ---------------------------------------------------------------------- #
    def do_create_room(self, line):
        'Create FS room in server with specific command (command, psw): ' \
            'create_room --cfg ../../federatedscope/example_configs' \
            '/distributed_femnist_server.yaml 12345'
        try:
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
        except Exception as error:
            print(f"Exception: {error}")

    def do_update_room(self, line):
        'Fetch all FS rooms from Lobby: update_room'
        try:
            global organizer
            result = organizer.send_task('server.display_room')
            cnt = 0
            while (not result.ready()) and cnt < self.timeout:
                print('Waiting for response... (will re-try in 1s)')
                time.sleep(1)
                cnt += 1
            self.room = result.get(timeout=1)
            info = ""
            for key, value in self.room.items():
                tmp = f"room_id: {key}, info: {value}\n"
                info += tmp
            print(info)
        except Exception as error:
            print(f"Exception: {error}")

    def do_view_room(self, line):
        'View specific FS room (room_id, psw): view_room 0 12345'
        try:
            global organizer
            room_id, psw = line.split(' ')
            result = organizer.send_task('server.view_room', [room_id, psw])
            cnt = 0
            while (not result.ready()) and cnt < self.timeout:
                print('Waiting for response... (will re-try in 1s)')
                time.sleep(1)
                cnt += 1
            print(result.get(timeout=1))
        except Exception as error:
            print(f"Exception: {error}")

    def do_quit(self, line):
        return True


if __name__ == '__main__':
    OrganizerClient().cmdloop()
