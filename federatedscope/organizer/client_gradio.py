import time
import gradio as gr

from celery import Celery

from federatedscope.core.configs.config import global_cfg
from federatedscope.organizer.cfg_client import server_ip
from federatedscope.organizer.utils import SSHManager, config2cmdargs, \
    flatten_dict, OrganizerLogger

TIMEOUT = 30
MAX_INFO_LEN = 30

# Rules: Naming Components `tab1_tab2_label_module`


# TODO: fix return value
# TODO: how to log information? `gr.Error`?
# TODO: room???
class UIEventHandler:
    def __init__(self):
        # TODO: make ecs_dict as ECS Manager
        self.ecs_dict = dict()
        # TODO: make ecs_dict as Task Manager
        self.task_dict = dict()
        self.organizer = Celery()
        self.organizer.config_from_object('cfg_client')
        self.logger = OrganizerLogger()

    def handle_display_ecs(self, value):
        info = ""
        for k, v in self.ecs_dict.items():
            info += f"ecs: {k}, info: {v}\n"
        if info:
            self.logger.info(info)
        else:
            self.logger.error('No saved ECS, please add ECS via `AddECS` '
                              'first!')
        return True

    def handle_add_ecs(self, ip, user, psw):
        key = f"{ip}"
        if key in self.ecs_dict:
            raise ValueError(f"ECS `{key}` already exists.")
        self.ecs_dict[key] = SSHManager(ip, user, psw)
        self.logger.info(f"{self.ecs_dict[key]} added.")

        return f'Successfully added {ip}!'

    def handle_del_ecs(self, value):
        key = value['ip']
        del self.ecs_dict[key]
        self.logger.info(f"Delete {key}: {self.ecs_dict[key]}.")
        # TODO: Del all task
        ...
        return True

    def handle_display_task(self, value):
        # TODO: add abort, check status, etc
        for i in self.ecs_dict:
            self.logger.info(f'{self.ecs_dict[i].ip}:'
                             f' {self.ecs_dict[i].tasks}')
        return True

    def handle_join_task(self, value):
        ip, task_id, yaml, opts = value['ip_join_task'], \
                                  value['task_id_join_task'], \
                                  value['yaml_join_task'], \
                                  value['opts_join_task']
        ecs, task = self.ecs_dict[ip], self.task_dict[task_id]

        if task['cfg'] == '******':
            # No access authority to the task.
            raise ValueError('Please get access authority via `access_task` '
                             'before joining.')

        cfg = global_cfg.clone()
        task_cfg = task['cfg']
        cfg.merge_from_list(task_cfg)

        # Merge other opts and convert to command string
        if yaml.endswith('.yaml'):
            cfg.merge_from_file(yaml)
        else:
            self.logger.warning('The yaml file is none or invalid, ignored.')
        opts = opts.split(' ')
        cfg.merge_from_list(opts)

        # Convert necessary configurations
        cfg['distribute']['server_host'] = server_ip
        cfg['distribute']['client_host'] = ip
        cfg['distribute']['role'] = 'client'

        cfg = config2cmdargs(flatten_dict(cfg))
        command = ''
        for i in cfg:
            value = f'{i}'.replace(' ', '')
            command += f' "{value}"'
        command = command[1:]
        pid = ecs.launch_task(command)
        self.logger.info(f'{ecs.ip}({pid}) launched,')
        return True

    def handle_create_task(self, yaml, opts, password=123456, **kwargs):
        if len(data) == 0:
            gr.Error("Data is invalid!")
        opts = opts.split(' ')
        cfg = global_cfg.clone()
        if yaml.endswith('.yaml'):
            cfg.merge_from_file(yaml)
        else:
            self.logger.warning('The yaml file is none or invalid, ignored.')
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
            self.logger.info('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        self.logger.info(result.get(timeout=1))
        return True

    def handle_update_task(self, value):
        self.logger.warning('Forget all saved room due to `update_room`.')
        result = self.organizer.send_task('server.display_room')
        cnt = 0
        while (not result.ready()) and cnt < TIMEOUT:
            self.logger.info('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        self.task_dict = result.get(timeout=1)
        if len(self.task_dict) == 0:
            self.logger.info(
                'No task available now. Please create a new task.')
            return False
        info = ""
        for k, v in self.task_dict.items():
            tmp = f"room_id: {k}, info: {v}\n"
            info += tmp
        self.logger.info(info)
        return True

    def handle_access_task(self, value):
        task_id, psw, verbose = value['task_id_access_task'], \
                                value['password_access_task'], \
                                value['vb_access_task']
        result = self.organizer.send_task('server.view_room', [task_id, psw])
        cnt = 0
        while (not result.ready()) and cnt < TIMEOUT:
            self.logger.info('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        info = result.get(timeout=1)
        if isinstance(info, dict):
            # Get authority,
            # verbose 0: print no information
            # verbose 1: print information of a specific room
            # verbose 2: print information of all the rooms
            self.task_dict[task_id] = info
            self.logger.info(
                f'Task {task_id} has been updated to be join-able.')
            if verbose == '1':
                self.logger.info(info)
            elif verbose == '2':
                self.logger.info(self.task_dict)
            else:
                self.logger.info(f'Authority of task {task_id} get!')
        else:
            # Authority denied
            self.logger.info(info)
        return True

    def handle_shut_down(self):
        result = self.organizer.send_task('server.shut_down')
        cnt = 0
        while (not result.ready()) and cnt < TIMEOUT:
            self.logger.info('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        self.logger.info(result.get(timeout=1))
        return 'Shut down all tasks successfully.'


handler = UIEventHandler()

with gr.Blocks() as demo:
    gr.Markdown("Welcome to FederatedScope Cloud Demo!")

    with gr.Tab("Task Runner"):
        with gr.Box():  # "Basic settings"
            gr.Markdown("Basic settings")
            with gr.Row():
                with gr.Box():
                    data_text = gr.Markdown("Data")
                    with gr.Tab("Choose"):
                        data = gr.Dropdown(
                            ['adult', 'credit', 'abalone', 'blog'],
                            value=['adult'],
                            label='vFL data')
                    with gr.Tab("Upload"):
                        data_text = gr.Markdown("Data")
                        data_upload_button = gr.UploadButton(
                            "Click to Upload "
                            "data",
                            file_types=['file'],
                            file_count="single")
                opts = gr.Textbox(label='Opts')
        with gr.Box():  # "Tuning setting"
            gr.Markdown("Tuning setting")
            optimizer = gr.Dropdown(['rs', 'bo_rf', 'bo_gp'],
                                    value=['bo_rf'],
                                    label='Optimizer')
            with gr.Row():
                model = gr.Dropdown(['lr', 'xgb', 'gbdt'],
                                    value=['lr', 'xgb'],
                                    multiselect=True,
                                    label='Model Selection')
                feat = gr.Dropdown([
                    '', 'min_max_norm', 'instance_norm', 'standardization',
                    'log_transform', 'uniform_binning', 'variance_filter',
                    'iv_filter'
                ],
                                   value=[
                                       '', 'min_max_norm', 'instance_norm',
                                       'standardization', 'log_transform',
                                       'uniform_binning', 'variance_filter',
                                       'iv_filter'
                                   ],
                                   multiselect=True,
                                   label='Feature Engineer')
            with gr.Box():
                lr = gr.Markdown("Learning Rate")
                with gr.Row():
                    min_lr = gr.Slider(0, 1, value=0.1, label='Minimum')
                    max_lr = gr.Slider(0, 1, value=0.8, label='Maximum')
            with gr.Box():
                yaml = gr.Markdown("Yaml (optional)")
                upload_button = gr.UploadButton("Click to Upload a Yaml",
                                                file_types=['file'],
                                                file_count="single")
            upload_button.upload(None, upload_button, None)

        # with gr.Row():
        #     task_input = [gr.Textbox(label='Yaml'), gr.Textbox(label='Opts')]

        task_input = [data, opts, model, feat, min_lr, max_lr, optimizer]
        task_button = gr.Button("Launch")

    with gr.Tab("ECS Manager"):
        with gr.Row():
            ecs_input = [
                gr.Textbox(label='IP'),
                gr.Textbox(label='User'),
                gr.Textbox(label='Password', type='password')
            ]
        ecs_button = gr.Button("Add")

    output = gr.Textbox(label='Logs', lines=10, interactive=False)

    shutdown_button = gr.Button("Shutdown")

    # Event
    task_button.click(handler.handle_create_task,
                      inputs=task_input,
                      outputs=output)
    ecs_button.click(handler.handle_add_ecs, inputs=ecs_input, outputs=output)
    shutdown_button.click(handler.handle_shut_down,
                          inputs=None,
                          outputs=output)

demo.launch(share=False, server_name="0.0.0.0", debug=True, server_port=7860)
