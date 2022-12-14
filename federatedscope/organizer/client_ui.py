"""
To conduct this on your mac, please execute:
```python
conda create -n fs_client python=3.10
conda activate fs_client
pip install -e .
conda install -c anaconda python.app
pythonw client_ui.py
```
"""

# TODO: delete this, cannot
import time
import datetime

from celery import Celery
from gooey import Gooey, GooeyParser

from federatedscope.core.configs.config import CN
from federatedscope.organizer.cfg_client import server_ip
from federatedscope.organizer.utils import SSHManager, config2cmdargs, \
    flatten_dict


def format_print(s):
    print(f"[{str(datetime.datetime.now()).split('.')[0]}] - {s}")


@Gooey(richtext_controls=True,
       program_name="FederatedScopeCloudOrganizer",
       encoding="utf-8",
       progress_regex=r"^progress: (\d+)%$",
       return_to_config=True)
def main():
    organizer = Celery()
    organizer.config_from_object('cfg_client')
    # ---------------------------------------------------------------------- #
    # Global variables
    # ---------------------------------------------------------------------- #
    ecs_dict, room_dict = dict(), dict()
    timeout = 10

    settings_msg = 'FederatedScope Cloud Organizer, powered by FS team.'
    parser = GooeyParser(description=settings_msg)

    subs = parser.add_subparsers(help='commands', dest='command')

    # ---------------------------------------------------------------------- #
    # ECS Manager
    # ---------------------------------------------------------------------- #
    Leftparser = subs.add_parser('ECS_Manager')
    Leftparser.add_argument("--DisplayEcs",
                            metavar='Display saved ECS in client control list',
                            widget='BlockCheckbox',
                            action="store_true",
                            default=True)

    # do_add_ecs
    Leftparser = subs.add_parser('__Add_ECS')
    Leftparser.add_argument("ip", metavar='ip address', default='172.X.X.X')
    Leftparser.add_argument("user", metavar='user name of ECS', default='root')
    Leftparser.add_argument("psw",
                            metavar='password of user',
                            default='123456',
                            widget='PasswordField')

    # do_del_ecs
    Leftparser = subs.add_parser('__Del_ECS')
    Leftparser.add_argument("ip", metavar='ip address', default='172.X.X.X')

    # do_join_room
    Leftparser = subs.add_parser('__Join_Room')
    Leftparser.add_argument("ip", metavar='ip address', default='172.X.X.X')
    Leftparser.add_argument("room_id", metavar='room id', default='0')
    Leftparser.add_argument("cfg",
                            metavar='cfg file',
                            default='test.yaml',
                            widget='FileChooser')
    Leftparser.add_argument("opt", metavar='other opts', default='device 0')

    # ---------------------------------------------------------------------- #
    # Task manager related
    # ---------------------------------------------------------------------- #
    Leftparser = subs.add_parser('Room_Manager')
    Leftparser.add_argument("--DisplayTask",
                            metavar='Display all running tasks in client '
                            'task list',
                            widget='BlockCheckbox',
                            action="store_true",
                            default=True)

    # do_create_room
    Leftparser = subs.add_parser('__Create_Room')
    Leftparser.add_argument("cfg",
                            metavar='cfg file',
                            default='test.yaml',
                            widget='FileChooser')

    # do_update_room
    Leftparser = subs.add_parser('__Update_Room')
    Leftparser.add_argument("--UpdateRoom",
                            metavar='Fetch all FS rooms from Lobby (will '
                            'forget all saved room)',
                            widget='BlockCheckbox',
                            action="store_true",
                            default=True)

    # do_access_room
    Leftparser = subs.add_parser('__Access_Room')
    Leftparser.add_argument("id", metavar='room id', default='0')
    Leftparser.add_argument("psw", metavar='password', default='123456')
    Leftparser.add_argument("verbose",
                            metavar='verbose',
                            widget="Listbox",
                            nargs="*",
                            choices=[0, 1, 2],
                            default=[0],
                            help='    0: print no information\n'
                            '    1: print information of a specific room\n'
                            '    2: print information of all the rooms')
    # ---------------------------------------------------------------------- #
    # Server manager related
    # ---------------------------------------------------------------------- #
    Leftparser = subs.add_parser('Server_Manager')
    Leftparser.add_argument("--ShutDown",
                            metavar='Shut down all rooms and quit',
                            widget='BlockCheckbox',
                            action="store_true",
                            default=True)

    args = parser.parse_args()

    if args.command == 'ECS_Manager':
        try:
            info = ""
            for key, value in ecs_dict.items():
                info += f"ecs: {key}, info: {value}\n"
            if info:
                format_print(info)
            else:
                format_print('No saved ECS, please add ECS via `__Add_ECS` '
                             'first!')
        except Exception as error:
            format_print(f"Exception: {error}")
    elif args.command == '__Add_ECS':
        try:
            ip, user, psw = args.ip, args.user, args.psw
            key = f"{ip}"
            if key in ecs_dict:
                raise ValueError(f"ECS `{key}` already exists.")
            ecs_dict[key] = SSHManager(ip, user, psw)
            format_print(f"{ecs_dict[key]} added.")
        except Exception as error:
            format_print(f"Exception: {error}")
    elif args.command == '__Del_ECS':
        try:
            ip = args.ip
            format_print(f"Delete {ip}: {ecs_dict[ip]}.")
            # TODO: Del all task
            del ecs_dict[ip]
        except Exception as error:
            format_print(f"Exception: {error}")
    elif args.command == '__Join_Room':
        try:
            ...
        except Exception as error:
            format_print(f"Exception: {error}")
    elif args.command == 'Room_Manager':
        try:
            ...
        except Exception as error:
            format_print(f"Exception: {error}")
    elif args.command == '__Create_Room':
        try:
            ...
        except Exception as error:
            format_print(f"Exception: {error}")
    elif args.command == '__Update_Room':
        try:
            ...
        except Exception as error:
            format_print(f"Exception: {error}")
    elif args.command == '__Access_Room':
        try:
            ...
        except Exception as error:
            format_print(f"Exception: {error}")
    elif args.command == 'Server_Manager':
        try:
            ...
        except Exception as error:
            format_print(f"Exception: {error}")
    else:
        print("Thanks for using FederatedScopeCloudOrganizer, bye!")
        time.sleep(1)
        return


if __name__ == '__main__':
    main()
