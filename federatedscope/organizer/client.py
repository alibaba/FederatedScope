import cmd2
import time
from celery import Celery
from cmd2 import Bg, Fg, style

from federatedscope.core.configs.config import CN
from federatedscope.organizer.cfg_client import server_ip
from federatedscope.organizer.utils import SSHManager, config2cmdargs, \
    flatten_dict

organizer = Celery()
organizer.config_from_object('cfg_client')


class OrganizerClient(cmd2.Cmd):
    SEVER_CATEGORY = 'Server Related Commands'
    ECS_CATEGORY = 'ECS Related Commands'
    TASK_CATEGORY = 'Task Related Commands'

    # Maintained several dict
    ecs_dict, room_dict = {}, {}
    timeout = 10

    def __init__(self):
        super().__init__(
            multiline_commands=['echo'],
            include_ipy=True,
        )

        self.intro = style(
            'Welcome to the FS organizer shell. Type help or ? to list '
            'commands.\n',
            fg=Fg.BLUE,
            bg=Bg.WHITE,
            bold=True)
        self.prompt = 'FederatedScope>> '
        self.self_in_py = True
        self.default_category = 'Built-in Commands'
        self.debug = True
        self.foreground_color = Fg.CYAN.name.lower()

    def fancy_output(self, out):
        return self.poutput(style(out, fg=Fg.GREEN, bg=Bg.WHITE))

    # ---------------------------------------------------------------------- #
    # SSH Manager related
    # ---------------------------------------------------------------------- #
    @cmd2.with_category(ECS_CATEGORY)
    def do_add_ecs(self, line):
        'Usage: add_ecs ip user psw\n\n' \
            'Add ECS to client control list\n\n' \
            'required arguments:\n' \
            '   ip, ip address 172.X.X.X\n' \
            '   user, user name of ECS\n' \
            '   psw, password of user\n\n' \
            'Example:\n' \
            '   add_ecs 172.X.X.X root 12345\n'
        try:
            ip, user, psw = line.split(' ')
            key = f"{ip}"
            if key in self.ecs_dict:
                raise ValueError(f"ECS `{key}` already exists.")
            self.ecs_dict[key] = SSHManager(ip, user, psw)
            self.fancy_output(f"{self.ecs_dict[key]} added.")
        except Exception as error:
            self.pexcept(f"Exception: {error}")

    @cmd2.with_category(ECS_CATEGORY)
    def do_del_ecs(self, line):
        'Usage: del_ecs ip\n\n' \
            'Delete ECS from client control list\n\n' \
            'required arguments:\n' \
            '   ip, ip address 172.X.X.X\n\n' \
            'Example:\n' \
            '   del_ecs 172.X.X.X\n'
        try:
            key = line
            self.fancy_output(f"Delete {key}: {self.ecs_dict[key]}.")
            # TODO: Del all task
            del self.ecs_dict[key]
        except Exception as error:
            self.pexcept(f"Exception: {error}")

    @cmd2.with_category(ECS_CATEGORY)
    def do_display_ecs(self, line):
        'Usage: display_ecs' \
            'Display saved ECS in client control list\n\n' \
            'Example:\n' \
            '   display_ecs\n'
        try:
            info = ""
            for key, value in self.ecs_dict.items():
                info += f"ecs: {key}, info: {value}\n"
            self.fancy_output(info)
        except Exception as error:
            self.pexcept(f"Exception: {error}")

    @cmd2.with_category(ECS_CATEGORY)
    def do_join_room(self, line):
        'Usage: join_room ip room_id other_opts\n\n' \
            'Let an ECS join a specific room\n\n' \
            'required arguments:\n' \
            '   ip, ip address 172.X.X.X\n' \
            '   room_id, room id joining \n' \
            '   other_opts, other operations in FS\n\n' \
            'Example:\n' \
            '   join_room 172.X.X.X 0 device 0 distribute.data_idx 2 ...\n'
        try:
            line = line.split(' ')
            ip, room_id, opts = line[0], line[1], line[2:]
            ecs, room = self.ecs_dict[ip], self.room_dict[room_id]
            if room['cfg'] == '******':
                raise ValueError('Please view room before joining.')
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
            pid = ecs.launch_task(command)
            self.fancy_output(f'{ecs.ip}({pid}) launched,')
        except Exception as error:
            self.pexcept(f"Exception: {error}")

    # ---------------------------------------------------------------------- #
    # Task manager related
    # ---------------------------------------------------------------------- #
    @cmd2.with_category(TASK_CATEGORY)
    def do_display_task(self, line):
        'Usage: display_task' \
            'Display all running tasks in client task list\n\n' \
            'Example:\n' \
            '   display_task\n'
        # TODO: add abort, check status, etc
        for i in self.ecs_dict:
            self.fancy_output(f'{self.ecs_dict[i].ip}:'
                              f' {self.ecs_dict[i].tasks}')

    # ---------------------------------------------------------------------- #
    # Server related messages
    # ---------------------------------------------------------------------- #
    @cmd2.with_category(SEVER_CATEGORY)
    def do_create_room(self, line):
        'Usage: create_room command psw\n\n' \
            'Create FS room in server with specific command\n\n' \
            'required arguments:\n' \
            '   command, extra command to launch FS\n' \
            '   psw, password for room \n\n' \
            'Example:\n' \
            '   create_room --cfg ../../federatedscope/example_configs' \
            '/distributed_femnist_server.yaml 12345\n'
        try:
            global organizer
            psw = line.split(' ')[-1]
            command = line[:-len(psw) - 1]
            result = organizer.send_task('server.create_room', [command, psw])
            cnt = 0
            while (not result.ready()) and cnt < self.timeout:
                self.fancy_output(
                    'Waiting for response... (will re-try in 1s)')
                time.sleep(1)
                cnt += 1
            self.fancy_output(result.get(timeout=1))
        except Exception as error:
            self.pexcept(f"Exception: {error}")

    @cmd2.with_category(SEVER_CATEGORY)
    def do_update_room(self, line):
        'Usage: update_room' \
            'Fetch all FS rooms from Lobby (will forget all saved room)\n\n' \
            'Example:\n' \
            '   update_room\n'
        try:
            global organizer
            self.fancy_output('Forget all saved room due to `update_room`.')
            result = organizer.send_task('server.display_room')
            cnt = 0
            while (not result.ready()) and cnt < self.timeout:
                self.fancy_output(
                    'Waiting for response... (will re-try in 1s)')
                time.sleep(1)
                cnt += 1
            self.room_dict = result.get(timeout=1)
            info = ""
            for key, value in self.room_dict.items():
                tmp = f"room_id: {key}, info: {value}\n"
                info += tmp
            self.fancy_output(info)
        except Exception as error:
            self.pexcept(f"Exception: {error}")

    @cmd2.with_category(SEVER_CATEGORY)
    def do_view_room(self, line):
        'Usage: view_room room_id psw verbose\n\n' \
            'View specific FS room\n\n' \
            'required arguments:\n' \
            '   room_id, extra command to launch FS\n' \
            '   psw, password for room \n' \
            '   verbose,\n' \
            '       0: print no information\n' \
            '       1: print information of a specific room\n' \
            '       2: print information of all the rooms\n\n' \
            'Example:\n' \
            '   view_room 0 12345 0\n'
        try:
            global organizer
            room_id, psw, verbose = line.split(' ')
            result = organizer.send_task('server.view_room', [room_id, psw])
            cnt = 0
            while (not result.ready()) and cnt < self.timeout:
                self.fancy_output(
                    'Waiting for response... (will re-try in 1s)')
                time.sleep(1)
                cnt += 1
            info = result.get(timeout=1)
            if isinstance(info, dict):
                self.room_dict[room_id] = info
                self.fancy_output(
                    f'Room {room_id} has been updated to joinable.')
                if verbose == '1':
                    self.fancy_output(info)
                elif verbose == '2':
                    self.fancy_output(self.room_dict)
            else:
                self.fancy_output(info)
        except Exception as error:
            self.pexcept(f"Exception: {error}")

    @cmd2.with_category(SEVER_CATEGORY)
    def do_shut_down(self, line):
        'Usage: shut_down' \
            'Shut down all rooms and quit\n\n' \
            'Example:\n' \
            '   shut_down\n'
        global organizer
        result = organizer.send_task('server.shut_down')
        cnt = 0
        while (not result.ready()) and cnt < self.timeout:
            self.fancy_output('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        self.fancy_output(result.get(timeout=1))
        return True


if __name__ == '__main__':
    OrganizerClient().cmdloop()
