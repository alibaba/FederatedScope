# import time
#
# import gradio as gr
# from celery import Celery
#
# from federatedscope.core.configs.config import global_cfg
# from federatedscope.organizer.cfg_client import server_ip
# from federatedscope.organizer.utils import OrganizerLogger,
# config2cmdargs, flatten_dict
# from federatedscope.organizer.module.ssh import SSHManager
#
# TIMEOUT = 30
# MAX_INFO_LEN = 30
#
#
# class EventHandler:
#     def __init__(self):
#         # TODO: make ecs_dict as ECS Manager
#         self.ecs_dict = dict()
#         # TODO: make ecs_dict as Task Manager
#         self.task_dict = dict()
#         self.organizer = Celery()
#         self.organizer.config_from_object('cfg_client')
#         self.logger = OrganizerLogger()
#
#     def handle_display_ecs(self, value):
#         info = ""
#         for k, v in self.ecs_dict.items():
#             info += f"ecs: {k}, info: {v}\n"
#         if info:
#             self.logger.info(info)
#         else:
#             self.logger.error('No saved ECS, please add ECS via `AddECS` '
#                               'first!')
#         return True
#
#     def handle_add_ecs(self, ip, user, psw):
#         key = f"{ip}"
#         if key in self.ecs_dict:
#             raise ValueError(f"ECS `{key}` already exists.")
#         self.ecs_dict[key] = SSHManager(ip, user, psw)
#         self.logger.info(f"{self.ecs_dict[key]} added.")
#
#         return f'Successfully added {ip}!'
#
#     def handle_del_ecs(self, value):
#         key = value['ip']
#         del self.ecs_dict[key]
#         self.logger.info(f"Delete {key}: {self.ecs_dict[key]}.")
#         # TODO: Del all task
#         ...
#         return True
#
#     def handle_display_task(self, value):
#         # TODO: add abort, check status, etc
#         for i in self.ecs_dict:
#             self.logger.info(f'{self.ecs_dict[i].ip}:'
#                              f' {self.ecs_dict[i].tasks}')
#         return True
#
#     def handle_join_task(self, value):
#         ip, task_id, yaml, opts = value['ip_join_task'], \
#                                   value['task_id_join_task'], \
#                                   value['yaml_join_task'], \
#                                   value['opts_join_task']
#         ecs, task = self.ecs_dict[ip], self.task_dict[task_id]
#
#         if task['cfg'] == '******':
#             # No access authority to the task.
#             raise ValueError('Please get access authority via `access_task` '
#                              'before joining.')
#
#         cfg = global_cfg.clone()
#         task_cfg = task['cfg']
#         cfg.merge_from_list(task_cfg)
#
#         # Merge other opts and convert to command string
#         if yaml.endswith('.yaml'):
#             cfg.merge_from_file(yaml)
#         else:
#             self.logger.warning('The yaml file is none or invalid, ignored.')
#         opts = opts.split(' ')
#         cfg.merge_from_list(opts)
#
#         # Convert necessary configurations
#         cfg['distribute']['server_host'] = server_ip
#         cfg['distribute']['client_host'] = ip
#         cfg['distribute']['role'] = 'client'
#
#         cfg = config2cmdargs(flatten_dict(cfg))
#         command = ''
#         for i in cfg:
#             value = f'{i}'.replace(' ', '')
#             command += f' "{value}"'
#         command = command[1:]
#         pid = ecs.launch_task(command)
#         self.logger.info(f'{ecs.ip}({pid}) launched,')
#         return True
#
#     def handle_create_task(self, yaml, opts, password=123456, **kwargs):
#         if len(data) == 0:
#             gr.Error("Data is invalid!")
#         opts = opts.split(' ')
#         cfg = global_cfg.clone()
#         if yaml.endswith('.yaml'):
#             cfg.merge_from_file(yaml)
#         else:
#             self.logger.warning('The yaml file is none or invalid, ignored.')
#         cfg.merge_from_list(opts)
#         cfg = config2cmdargs(flatten_dict(cfg))
#
#         command = ''
#         for i in cfg:
#             value = f'{i}'.replace(' ', '')
#             command += f' "{value}"'
#         command = command[1:]
#         result = self.organizer.send_task('server.create_room',
#                                           [command, password])
#         cnt = 0
#         while (not result.ready()) and cnt < TIMEOUT:
#             self.logger.info('Waiting for response... (will re-try in 1s)')
#             time.sleep(1)
#             cnt += 1
#         self.logger.info(result.get(timeout=1))
#         return True
#
#     def handle_update_task(self, value):
#         self.logger.warning('Forget all saved room due to `update_room`.')
#         result = self.organizer.send_task('server.display_room')
#         cnt = 0
#         while (not result.ready()) and cnt < TIMEOUT:
#             self.logger.info('Waiting for response... (will re-try in 1s)')
#             time.sleep(1)
#             cnt += 1
#         self.task_dict = result.get(timeout=1)
#         if len(self.task_dict) == 0:
#             self.logger.info(
#                 'No task available now. Please create a new task.')
#             return False
#         info = ""
#         for k, v in self.task_dict.items():
#             tmp = f"room_id: {k}, info: {v}\n"
#             info += tmp
#         self.logger.info(info)
#         return True
#
#     def handle_access_task(self, value):
#         task_id, psw, verbose = value['task_id_access_task'], \
#                                 value['password_access_task'], \
#                                 value['vb_access_task']
#         result = self.organizer.send_task('server.view_room', [task_id, psw])
#         cnt = 0
#         while (not result.ready()) and cnt < TIMEOUT:
#             self.logger.info('Waiting for response... (will re-try in 1s)')
#             time.sleep(1)
#             cnt += 1
#         info = result.get(timeout=1)
#         if isinstance(info, dict):
#             # Get authority,
#             # verbose 0: print no information
#             # verbose 1: print information of a specific room
#             # verbose 2: print information of all the rooms
#             self.task_dict[task_id] = info
#             self.logger.info(
#                 f'Task {task_id} has been updated to be join-able.')
#             if verbose == '1':
#                 self.logger.info(info)
#             elif verbose == '2':
#                 self.logger.info(self.task_dict)
#             else:
#                 self.logger.info(f'Authority of task {task_id} get!')
#         else:
#             # Authority denied
#             self.logger.info(info)
#         return True
#
#     def handle_shut_down(self):
#         result = self.organizer.send_task('server.shut_down')
#         cnt = 0
#         while (not result.ready()) and cnt < TIMEOUT:
#             self.logger.info('Waiting for response... (will re-try in 1s)')
#             time.sleep(1)
#             cnt += 1
#         self.logger.info(result.get(timeout=1))
#         return 'Shut down all tasks successfully.'
