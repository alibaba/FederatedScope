import time

from utils.utils import check_file
from cfg import get_fl_cfg

import subprocess
import argparse
import os

MAX_NUMBER_CLIENT_PER_MACHINE = 150
HINT_LENGTH = 60
AVD_NAME = "test_x86"
DEBUG = False
NO_DEBUG_SUFFIX = " >/dev/null 2>&1"


def remote_cmd(host, user, passwd, cmd, replace=True, feedback=False, print_log=False):
    if replace:
        command = cmd.replace(" ", "\ ")
    else:
        command = cmd

    execute_cmd = f"./utils/ssh_cmd.sh {host} {user} {passwd} {command}"
    if feedback:
        return subprocess.getoutput(execute_cmd)
    else:
        if DEBUG or print_log:
            os.system(execute_cmd)
        else:
            os.system(execute_cmd + NO_DEBUG_SUFFIX)


def remote_scp(host, user, passwd, files, target_dir, print_log=False):
    if isinstance(files, list):
        files = "\ ".join(files)
    elif isinstance(files, str):
        files = files.replace(" ", "\ ")
    else:
        raise TypeError(f"Type of files {type(files)} is wrong, str or list is expected")

    execute_cmd = f"./utils/scp_file.sh {host} {user} {passwd} {files} {target_dir}"
    if DEBUG or print_log:
        os.system(execute_cmd)
    else:
        os.system(execute_cmd + NO_DEBUG_SUFFIX)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=str, help="path of configuration")
    parser.add_argument("--reboot", type=bool, default=False)
    args = parser.parse_args()

    # If exists
    if not os.path.exists(args.cfg):
        print(f"Configuration {args.cfg} not exists")
        exit()

    # Load configuration
    cfg = get_fl_cfg()
    cfg.merge_from_file(args.cfg)

    ################################################################
    #                 STAGE 1: Local Preparation                   #
    ################################################################
    print("#" * HINT_LENGTH)
    print("#\033[1;34m" + "STAGE 1: Local Preparation".center(HINT_LENGTH - 2, " ") + "\033[0m#")
    print("#" * HINT_LENGTH)

    # Step 1: Check machine list
    print(f"Step 1: Checking machines ...")
    check_file(cfg.machine.host_path)
    machines = list()
    with open(cfg.machine.host_path) as file:
        for line in file:
            machine, cpu, memory = line.replace("\n", "").replace(" ", "").split(",")
            machines.append(machine)
    n_machines = len(machines)
    print(f"Done with {n_machines} machines")

    ################################################################
    #                  STAGE 2: Stop emulators                     #
    ################################################################
    print("#" * HINT_LENGTH)
    print("#\033[1;34m" + "STAGE 2: Stop emulators".center(HINT_LENGTH - 2, " ") + "\033[0m#")
    print("#" * HINT_LENGTH)

    for machine in machines:
        print(f"Mkdir in remote machine {cfg.machine.workdir} ...")
        remote_cmd(machine, cfg.machine.user, cfg.machine.passwd, f"mkdir -p {cfg.machine.workdir}")
        print("Done")

        print(f"Pass shell to machine {machine} ...")
        remote_scp(machine, cfg.machine.user, cfg.machine.passwd, "stop_emulator.sh", cfg.machine.workdir)
        print("Done")

        print(f"Kill emulators in remote machine {machine} ...")
        # obtain all emulators
        remote_cmd(machine, cfg.machine.user, cfg.machine.passwd, f"bash {cfg.machine.workdir}/stop_emulator.sh", feedback=True)

        time.sleep(3)

        remote_cmd(machine, cfg.machine.user, cfg.machine.passwd, "adb devices", print_log=True)

        print("Done")
        print("-" * HINT_LENGTH)

        if args.reboot:
            print(f"Reboot the machine {machine}")
            remote_cmd(machine, cfg.machine.user, cfg.machine.passwd, "reboot")
            print("Done")
