import cmd
import time
from celery import Celery

from federatedscope.core.configs.config import CN
from federatedscope.organizer.cfg_client import server_ip
from federatedscope.organizer.utils import SSHManager, config2cmdargs, \
    flatten_dict

organizer = Celery()
organizer.config_from_object('cfg_client')


class OrganizerClient(cmd.Cmd):
    intro = 'Welcome to the FS organizer shell. Type help or ? to list ' \
            'commands.\n'
    prompt = 'FederatedScope>> '
    # Maintained several dict
    ecs_dict, room_dict, task_dict = {}, {}, {}
    timeout = 10

    # ---------------------------------------------------------------------- #
    # SSH Manager related
    # ---------------------------------------------------------------------- #
    def do_add_ecs(self, line):
        'Add Ecs (ip, user, psw): add_ecs 172.X.X.X root 123'
        try:
            ip, user, psw = line.split(' ')
            key = f"{ip}"
            if key in self.ecs_dict:
                raise ValueError(f"ECS `{key}` already exists.")
            self.ecs_dict[key] = SSHManager(ip, user, psw)
            print(f"{self.ecs_dict[key]} added.")
        except Exception as error:
            print(f"Exception: {error}")

    def do_del_ecs(self, line):
        'Delete Ecs (ip): del_ecs 172.X.X.X'
        try:
            key = line
            print(f"Delete {key}: {self.ecs_dict[key]}.")
            del self.ecs_dict[key]
        except Exception as error:
            print(f"Exception: {error}")

    def do_display_ecs(self, line):
        'Display all saved ECS: display_ecs'
        try:
            info = ""
            for key, value in self.ecs_dict.items():
                info += f"ecs: {key}, info: {value}\n"
            print(info)
        except Exception as error:
            print(f"Exception: {error}")

    def do_join_room(self, line):
        'Let an ECS join a specific room (ip room_id other_opts): ' \
            'join_room 172.X.X.X 0 device 0 distribute.data_idx 2 ...'
        try:
            line = line.split(' ')
            ip, room_id, opts = line[0], line[1], line[2:]
            ecs, room = self.ecs_dict[ip], self.room_dict[room_id]
            cfg = CN(room['cfg'])

            # Convert necessary configurations
            cfg['distribute']['server_host'] = server_ip
            cfg['distribute']['client_host'] = ip
            cfg['distribute']['role'] = 'client'

            # Merge other opts and convert to command string
            cfg.merge_from_list(opts)
            cfg = config2cmdargs(flatten_dict(cfg))
            command = ''
            for i in cfg:
                value = f'{i}'.replace(' ', '')
                command += f' "{value}"'
            command = command[1:]
            # TODO: manage the process
            stdout, stderr = ecs.exec_python_cmd(f'nohup python '
                                                 f'federatedscope/main.py'
                                                 f' {command} &')
            print(stdout, stderr)
        except Exception as error:
            print(f"Exception: {error}")

    # ---------------------------------------------------------------------- #
    # Task manager related
    # ---------------------------------------------------------------------- #
    def do_display_task(self, line):
        # TODO: add abort, check status, etc
        print(self.task_dict)

    # ---------------------------------------------------------------------- #
    # Server related messages
    # ---------------------------------------------------------------------- #
    def do_create_room(self, line):
        'Create FS room in server with specific command (command, psw): ' \
            'create_room --cfg ../../federatedscope/example_configs' \
            '/distributed_femnist_server.yaml 123'
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
            print('Forget all saved room due to `update_room`.')
            result = organizer.send_task('server.display_room')
            cnt = 0
            while (not result.ready()) and cnt < self.timeout:
                print('Waiting for response... (will re-try in 1s)')
                time.sleep(1)
                cnt += 1
            self.room_dict = result.get(timeout=1)
            info = ""
            for key, value in self.room_dict.items():
                tmp = f"room_id: {key}, info: {value}\n"
                info += tmp
            print(info)
        except Exception as error:
            print(f"Exception: {error}")

    def do_view_room(self, line):
        'View specific FS room (room_id, psw, verbose): view_room 0 123 0\n' \
            'verbose 0: print no information\n' \
            'verbose 1: print information of a specific room\n' \
            'verbose 2: print information of all the rooms'
        try:
            global organizer
            room_id, psw, verbose = line.split(' ')
            result = organizer.send_task('server.view_room', [room_id, psw])
            cnt = 0
            while (not result.ready()) and cnt < self.timeout:
                print('Waiting for response... (will re-try in 1s)')
                time.sleep(1)
                cnt += 1
            info = result.get(timeout=1)
            if isinstance(info, dict):
                self.room_dict[room_id] = info
                print(f'Room {room_id} has been updated to joinable.')
                if verbose == '1':
                    print(info)
                elif verbose == '2':
                    print(self.room_dict)
            else:
                print(info)
        except Exception as error:
            print(f"Exception: {error}")

    def do_shut_down(self, line):
        'Shut down all rooms and quit: shut_down'
        global organizer
        result = organizer.send_task('server.shut_down')
        cnt = 0
        while (not result.ready()) and cnt < self.timeout:
            print('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        print(result.get(timeout=1))
        return True

    def do_quit(self, line):
        return True


if __name__ == '__main__':
    OrganizerClient().cmdloop()
