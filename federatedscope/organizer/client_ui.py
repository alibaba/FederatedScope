import PySimpleGUI as sg

import time

from celery import Celery

from federatedscope.core.configs.config import CN
from federatedscope.organizer.cfg_client import server_ip
from federatedscope.organizer.utils import SSHManager, config2cmdargs, \
    flatten_dict, format_print

TIMEOUT = 10


class UIEventHandler:
    def __init__(self):
        self.ecs_dict = dict()
        self.task_dict = dict()
        self.organizer = Celery()
        self.organizer.config_from_object('cfg_client')

    def handle_display_ecs(self, value):
        info = ""
        for k, v in self.ecs_dict.items():
            info += f"ecs: {k}, info: {v}\n"
        if info:
            format_print(info)
        else:
            format_print('No saved ECS, please add ECS via `AddECS` '
                         'first!')

    def handle_add_ecs(self, value):
        ip, user, psw = value['ip'], value['user'], value['password']
        key = f"{ip}"
        if key in self.ecs_dict:
            raise ValueError(f"ECS `{key}` already exists.")
        self.ecs_dict[key] = SSHManager(ip, user, psw)
        format_print(f"{self.ecs_dict[key]} added.")

    def handle_del_ecs(self, value):
        key = value['ip']
        del self.ecs_dict[key]
        format_print(f"Delete {key}: {self.ecs_dict[key]}.")
        # TODO: Del all task
        ...

    def handle_display_task(self, value):
        # TODO: add abort, check status, etc
        for i in self.ecs_dict:
            format_print(f'{self.ecs_dict[i].ip}:'
                         f' {self.ecs_dict[i].tasks}')

    def handle_join_task(self, value):
        ip, task_id, yaml, opts = value['ip'], value['task_id'], \
                                  value['yaml'], value['opts']
        ecs, task = self.ecs_dict[ip], self.task_dict[task_id]
        if task['cfg'] == '******':
            # No access to the task.
            raise ValueError('Please get access authority via `access_task` '
                             'before joining.')
        cfg = CN(task['cfg'])

        # Convert necessary configurations
        cfg['distribute']['server_host'] = server_ip
        cfg['distribute']['client_host'] = ip
        cfg['distribute']['role'] = 'client'

        # Merge other opts and convert to command string
        cfg.merge_from_file(yaml)
        cfg.merge_from_list(opts)
        cfg = config2cmdargs(flatten_dict(cfg))
        command = ''
        for i in cfg:
            value = f'{i}'.replace(' ', '')
            command += f' "{value}"'
        command = command[1:]
        pid = ecs.launch_task(command)
        format_print(f'{ecs.ip}({pid}) launched,')

    def handle_create_task(self, value):
        yaml, opts, password = value['yaml'], value['opts'], value['password']

        cfg = CN()
        cfg.merge_from_file(yaml)
        cfg.merge_from_list(opts)
        cfg = config2cmdargs(flatten_dict(cfg))

        command = ''
        for i in cfg:
            value = f'{i}'.replace(' ', '')
            command += f' "{value}"'
        command = command[1:]

        result = self.organizer.send_task('server.create_room',
                                          [command, password])
        cnt = 0
        while (not result.ready()) and cnt < TIMEOUT:
            self.fancy_output('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        self.fancy_output(result.get(timeout=1))

    def handle_update_task(self, value):
        format_print('Forget all saved room due to `update_room`.')
        result = self.organizer.send_task('server.display_room')
        cnt = 0
        while (not result.ready()) and cnt < TIMEOUT:
            self.fancy_output('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        self.task_dict = result.get(timeout=1)
        info = ""
        for k, v in self.task_dict.items():
            tmp = f"room_id: {k}, info: {v}\n"
            info += tmp
        format_print(info)

    def handle_access_task(self, value):
        task_id, psw, verbose = value['task_id'], value['password'], \
                                value['vb']
        result = self.organizer.send_task('server.view_room', [task_id, psw])
        cnt = 0
        while (not result.ready()) and cnt < TIMEOUT:
            format_print('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        info = result.get(timeout=1)
        if isinstance(info, dict):
            self.task_dict[task_id] = info
            format_print(f'Room {task_id} has been updated to be join-able.')
            if verbose == '1':
                format_print(info)
            elif verbose == '2':
                format_print(self.task_dict)
        else:
            format_print(info)

    def handle_shut_down(self, value):
        result = self.organizer.send_task('server.shut_down')
        cnt = 0
        while (not result.ready()) and cnt < TIMEOUT:
            format_print('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        format_print(result.get(timeout=1))


def check_value(value_dict):
    values = list(value_dict.values())
    cnt = 0
    for i in values:
        if i is True:
            cnt += 1

    if cnt == 1:
        return True
    else:
        return False


def FederatedScopeCloudOrganizer():
    # The main GUI of FederatedScopeCloudOrganizer
    sg.theme('Gray Gray Gray')
    a = 0

    # ---------------------------------------------------------------------- #
    # ECS Manager related
    # ---------------------------------------------------------------------- #
    display_ecs_layout = [[
        sg.Text('Display saved ECS in client control '
                'list.')
    ], [sg.Checkbox('Enabled', default=False, k='display_ecs')]]
    add_ecs_layout = [[sg.Text('Add ECS to client control list.')],
                      [sg.Checkbox('Enabled', default=False, k='add_ecs')],
                      [
                          sg.T('IP Address', size=(8, 1)),
                          sg.Input('172.X.X.X', key='ip', size=(35, 1))
                      ],
                      [
                          sg.T('User Name', size=(8, 1)),
                          sg.Input('root', key='user', size=(35, 1))
                      ],
                      [
                          sg.T('Password', size=(8, 1)),
                          sg.Input('123456',
                                   password_char='*',
                                   key='password',
                                   size=(35, 1))
                      ]]
    del_ecs_layout = [[sg.Text('Delete ECS from client control list.')],
                      [sg.Checkbox('Enabled', default=False, k='del_ecs')],
                      [sg.Text('IP Address'),
                       sg.Input(key='ip')]]
    ecs_group = sg.TabGroup([[
        sg.Tab('DisplayECS', display_ecs_layout),
        sg.Tab('AddECS', add_ecs_layout),
        sg.Tab('DelECS', del_ecs_layout)
    ]])
    ecs_layout = [[ecs_group]]

    # ---------------------------------------------------------------------- #
    # Task manager related
    # ---------------------------------------------------------------------- #
    display_task_layout = [[
        sg.Text('Display all running tasks in client '
                'task list.')
    ], [sg.Checkbox('Enabled', default=False, k='display_task')]]
    join_task_layout = [[sg.Text('Let an ECS join a specific task.')],
                        [sg.Checkbox('Enabled', default=False, k='join_task')],
                        [
                            sg.T('IP Address', size=(8, 1)),
                            sg.Input('172.X.X.X', key='ip', size=(35, 1))
                        ],
                        [
                            sg.T('Task ID', size=(8, 1)),
                            sg.Input('0', key='task_id', size=(35, 1))
                        ],
                        [
                            sg.T('YAML file', size=(8, 1)),
                            sg.Input('', key='yaml', size=(35, 1)),
                            sg.FilesBrowse()
                        ],
                        [
                            sg.T('Opts', size=(8, 1)),
                            sg.Input('device 0', key='opts', size=(35, 1))
                        ]]
    create_task_layout = [
        [sg.Text('Create FS task in server with specific '
                 'command.')],
        [sg.Checkbox('Enabled', default=False, k='create_task')],
        [
            sg.T('YAML file', size=(8, 1)),
            sg.Input('', key='yaml', size=(35, 1)),
            sg.FilesBrowse()
        ],
        [
            sg.T('Opts', size=(8, 1)),
            sg.Input('device 0', key='opts', size=(35, 1))
        ],
        [
            sg.T('Password', size=(8, 1)),
            sg.Input('123456', password_char='*', key='password', size=(35, 1))
        ],
    ]
    update_task_layout = [[
        sg.Text('Fetch all FS tasks from remote server (will '
                'forget all saved room).')
    ], [sg.Checkbox('Enabled', default=False, k='update_task')]]

    # TODO: merge with join
    access_task_layout = [
        [sg.Text('Obtain access to a specific task.')],
        [sg.Checkbox('Enabled', default=False, k='access_task')],
        [
            sg.T('TaskID', size=(8, 1)),
            sg.Input('0', key='task_id', size=(35, 1))
        ],
        [
            sg.T('Password', size=(8, 1)),
            sg.Input('123456', password_char='*', key='password', size=(35, 1))
        ],
        [sg.T('Verbose', size=(8, 1)),
         sg.Input('0', key='vb', size=(35, 1))]
    ]
    task_group = sg.TabGroup([[
        sg.Tab('DisplayTask', display_task_layout),
        sg.Tab('JoinTask', join_task_layout),
        sg.Tab('CreateTask', create_task_layout),
        sg.Tab('UpdateTask', update_task_layout),
        sg.Tab('AccessTask', access_task_layout)
    ]])
    task_layout = [[task_group]]

    # ---------------------------------------------------------------------- #
    # Server related messages
    # ---------------------------------------------------------------------- #
    shut_down_layout = [[sg.Text('Shut down all rooms and quit')],
                        [sg.Checkbox('Enabled', default=False, k='shut_down')]]

    server_group = sg.TabGroup([[sg.Tab('ShutDown', shut_down_layout)]])

    server_layout = [[server_group]]

    main_layout = [[
        sg.TabGroup([[
            sg.Tab('ECSManager', ecs_layout),
            sg.Tab('TaskManager', task_layout),
            sg.Tab('ServerManager', server_layout)
        ]])
    ]]

    # ---------------------------------------------------------------------- #
    # Main layout
    # ---------------------------------------------------------------------- #
    output_layout = [[sg.Text('INFO:')], [sg.Output(size=(100, 15))]]

    layout = [[sg.Text('FederatedScope Cloud Organizer, powered by FS team.')],
              [sg.Frame('Actions', layout=main_layout)], output_layout,
              [sg.Button('RUN', bind_return_key=True)]]

    window = sg.Window('FederatedScopeCloudOrganizer',
                       layout,
                       default_element_size=(30, 2),
                       finalize=True)

    handler = UIEventHandler()

    while True:
        try:
            event, values = window.read()
            if event == 'RUN':
                if not check_value(values):
                    format_print('There are no executable commands or '
                                 'multiple commands. Please check the '
                                 'CheckBox.')
                    continue

                for k, v in values.items():
                    if v is True:
                        if hasattr(handler, f'handle_{k}'):
                            getattr(handler, f'handle_{k}')(values)
                        else:
                            format_print(f'No valid handle_{k}.')

            elif event == sg.WIN_CLOSED:  # always,  always give a way out!
                break
        except Exception as error:
            format_print(f"Exception: {error}")

    window.close()


if __name__ == '__main__':
    FederatedScopeCloudOrganizer()
