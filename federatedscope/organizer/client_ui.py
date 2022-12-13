"""
To conduct this on your mac, please execute:
```python
conda install -c anaconda python.app
pythonw client_ui.py
```
"""
import time

from celery import Celery
from gooey import Gooey, GooeyParser


@Gooey(richtext_controls=True,
       program_name="FederatedScopeCloudOrganizer",
       encoding="utf-8",
       progress_regex=r"^progress: (\d+)%$")
def main():
    organizer = Celery()
    organizer.config_from_object('cfg_client')

    settings_msg = 'FederatedScope Cloud Organizer, powered by FS team.'
    parser = GooeyParser(description=settings_msg)

    subs = parser.add_subparsers(help='commands', dest='command')

    # ---------------------------------------------------------------------- #
    # SSH Manager
    # ---------------------------------------------------------------------- #
    Leftparser = subs.add_parser('SSH Manager')
    Leftparser.add_argument("DisplayEcs",
                            metavar='Display saved ECS in client control list',
                            widget='BlockCheckbox')

    # do_add_ecs
    Leftparser = subs.add_parser('    Add ECS')
    Leftparser.add_argument("ip", metavar='ip address', default='172.X.X.X')
    Leftparser.add_argument("user", metavar='user name of ECS', default='root')
    Leftparser.add_argument("psw",
                            metavar='password of user',
                            default='123456',
                            widget='PasswordField')

    # do_del_ecs
    Leftparser = subs.add_parser('    Del ECS')
    Leftparser.add_argument("ip", metavar='ip address', default='172.X.X.X')

    # do_join_room
    Leftparser = subs.add_parser('    Join Room')
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
    Leftparser = subs.add_parser('Task manager')
    Leftparser.add_argument("DisplayEcs",
                            metavar='Display all running tasks in client '
                            'task list',
                            widget='BlockCheckbox')

    # do_create_room
    Leftparser = subs.add_parser('    Create Room')
    Leftparser.add_argument("cfg",
                            metavar='cfg file',
                            default='test.yaml',
                            widget='FileChooser')

    # do_update_room
    Leftparser = subs.add_parser('    Update Room')
    Leftparser.add_argument("UpdateRoom",
                            metavar='Fetch all FS rooms from Lobby (will '
                            'forget all saved room)',
                            widget='BlockCheckbox')

    # do_access_room
    Leftparser = subs.add_parser('    Access Room')
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
    Leftparser = subs.add_parser('Server manager')
    Leftparser.add_argument("ShutDown",
                            metavar='Shut down all rooms and quit',
                            widget='BlockCheckbox')

    args = parser.parse_args()

    if args.command == "cmd1":
        ...
    elif args.command == "cmd2":
        ...
    elif args.command == "cmd3":
        ...
    elif args.command == "cmd4":
        ...
    else:
        print("Bye!")
        time.sleep(1)
        return


if __name__ == '__main__':
    main()
