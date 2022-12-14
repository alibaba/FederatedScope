import PySimpleGUI as sg

import time
import datetime

from celery import Celery


def format_print(s):
    print(f"[{str(datetime.datetime.now()).split('.')[0]}] - {s}")


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
    ], [sg.Checkbox('Checkbox', default=False, k='-CB-display_ecs')]]
    add_ecs_layout = [[sg.Text('Add ECS to client control list.')]]
    del_ecs_layout = [[sg.Text('Delete ECS from client control list.')]]
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
    ]]
    join_task_layout = [[sg.Text('Let an ECS join a specific task.')]]
    create_task_layout = [[
        sg.Text('Create FS task in server with specific '
                'command.')
    ]]
    update_task_layout = [[
        sg.Text('Fetch all FS rooms from Lobby (will '
                'forget all saved room).')
    ]]
    access_task_layout = [[sg.Text('Obtain access to a spesific task.')]]
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
    shut_down_layout = [[sg.Text('This is inside of a tab')]]

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
    output_layout = [[sg.Text('Status')], [sg.Output(size=(100, 15))]]

    layout = [[sg.Text('FederatedScope Cloud Organizer, powered by FS team.')],
              [sg.Frame('Actions', layout=main_layout)], output_layout,
              [sg.Button('RUN', bind_return_key=True)]]

    window = sg.Window('FederatedScopeCloudOrganizer',
                       layout,
                       default_element_size=(30, 2),
                       finalize=True)

    while True:
        event, values = window.read()
        format_print(event)
        format_print(values)
        if event == 'RUN':
            a += 1
            format_print(a)
        elif event == sg.WIN_CLOSED:  # always,  always give a way out!
            break

    window.close()


if __name__ == '__main__':
    FederatedScopeCloudOrganizer()
