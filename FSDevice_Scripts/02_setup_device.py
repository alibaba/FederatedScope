import time

import numpy as np

from multiprocessing import Process, Queue

from utils.multi_tqdm import handle_msg, MultiProcessBar
from utils.utils import check_file
from cfg import get_fl_cfg

import subprocess
import argparse
import os

MAX_NUMBER_CLIENT_PER_MACHINE = 180
MIN_NUMBER_CLIENT_PER_MACHINE = 10
HINT_LENGTH = 85
AVD_NAME = "test_x86"
DEBUG = False
NO_DEBUG_SUFFIX = " >/dev/null 2>&1"
PUB2PRI = False
NCOLS = 100

# dispatch the emulators to different machines according to their capacities
capacity = {

}

# mapping from public ip to private ip address
pub2pri = {

}


def draw_table(machines, cpus, memories, partitions, distribution):
    title = ["Machine", "#Cpu core", "Memory(GB)", "#Clients", "Distribution"]
    lens = [max([len(_) for _ in machines]) + 2] + [len(_)+2 for _ in title]

    if sum(lens) < HINT_LENGTH:
        lens = [len(_) for _ in np.array_split(range(HINT_LENGTH-len(lens)-1), len(title))]

    ruler = "+" + "+".join(["-"*_ for _ in lens]) + "+"

    print(ruler)
    content = ["Machine", "#Cpu core", "Memory",  "#Clients", "Distribution"]
    content = [c.center(l, " ") for c, l in zip(content, lens)]
    print("|" + "|".join(content) + "|")
    print(ruler)

    for machine, cpu, memory, n_device in zip(machines, cpus, memories, partitions):
        content = [machine, cpu, memory, str(n_device), distribution]
        content = [c.center(l, " ") for c, l in zip(content, lens)]
        print("|" + "|".join(content) + "|")
        print(ruler)


def get_partitions(n_client, n_machines, machines):
    if MIN_NUMBER_CLIENT_PER_MACHINE * n_machines >= n_client:
        partitions = [MIN_NUMBER_CLIENT_PER_MACHINE] * (n_client // MIN_NUMBER_CLIENT_PER_MACHINE)
        if n_client % MIN_NUMBER_CLIENT_PER_MACHINE > 0:
            partitions += [n_client % MIN_NUMBER_CLIENT_PER_MACHINE]
        print("The clients are grouped to different machines equally.")
    else:
        weight_sum = np.sum([capacity.get(_, 1) for _ in machines]) * 1.
        partitions = [int(capacity.get(_, 1) / weight_sum * n_client) for _ in machines]
        last_devices = n_client - np.sum(partitions)
        for ind in range(last_devices):
            partitions[ind] += 1

        while 0 in partitions:
            partitions.remove(0)
        print("The clients are grouped to different machines according to their capacities:")

    assert sum(partitions) == n_client
    return partitions


def remote_cmd(host, user, passwd, cmd, replace=True, feedback=False, print_log=False):
    if PUB2PRI:
        host = pub2pri[host]

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


def remote_cmd_wt(host, user, passwd, cmd, replace=True, feedback=False, print_log=False):
    if PUB2PRI:
        host = pub2pri[host]

    if replace:
        command = cmd.replace(" ", "\ ")
    else:
        command = cmd

    execute_cmd = f"./utils/ssh_cmd_wt.sh {host} {user} {passwd} {command}"
    if feedback:
        return subprocess.getoutput(execute_cmd)
    else:
        if DEBUG or print_log:
            os.system(execute_cmd)
        else:
            os.system(execute_cmd + NO_DEBUG_SUFFIX)


def remote_scp(host, user, passwd, files, target_dir, print_log=False):
    if PUB2PRI:
        host = pub2pri[host]

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


def local_cmd(execute_cmd):
    if DEBUG:
        os.system(execute_cmd)
    else:
        os.system(execute_cmd + NO_DEBUG_SUFFIX)


def get_machine_datasets(datasets, start_index, n_dataset):
    base_times = start_index // len(datasets)
    end_index = start_index + n_dataset - base_times * len(datasets)
    start_index = start_index % len(datasets)

    machine_datasets = list()
    while True:
        if end_index > len(datasets):
            machine_datasets += datasets[start_index:]
            start_index = 0
            end_index -= len(datasets)
        else:
            machine_datasets += datasets[start_index: end_index]
            break
    return machine_datasets


class KillEmuProcess(Process):
    def __init__(self, queue, cfg, machine):
        super(KillEmuProcess, self).__init__()
        self.cfg = cfg
        self.machine = machine
        self.queue = queue

    def run(self):
        # Mkdir in remote machine
        remote_cmd(self.machine, self.cfg.machine.user, self.cfg.machine.passwd, f"mkdir -p {self.cfg.machine.workdir}")
        self.queue.put(("update", [machine, 30]))

        # Pass shell to machine
        remote_scp(self.machine, self.cfg.machine.user, self.cfg.machine.passwd, "stop_emulator.sh",
                   self.cfg.machine.workdir)
        self.queue.put(("update", [machine, 60]))

        # Kill emulators in remote machine ...
        # obtain all emulators
        remote_cmd(self.machine, self.cfg.machine.user, self.cfg.machine.passwd,
                   f"bash {self.cfg.machine.workdir}/stop_emulator.sh",
                   feedback=True)
        self.queue.put(("update", [machine, 100]))


class StartProcess(Process):
    def __init__(self, cfg, queue, machine, n_device):
        super(StartProcess, self).__init__()
        self.cfg = cfg
        self.queue = queue
        self.machine = machine
        self.n_device = n_device
        self.print_log = DEBUG

    def run(self):
        cfg = self.cfg
        machine = self.machine
        n_device = self.n_device

        print(f"Starting {self.n_device} emulators and clients in {self.machine} ...")

        # Install APP
        # Pass start_emulator.py and start_client.py to remote machine
        remote_scp(self.machine, cfg.machine.user, cfg.machine.passwd, ["start_emulator.py", "start_client.py"],
                   cfg.machine.workdir, print_log=self.print_log)
        # self.queue.put(("update", [machine, 20]))

        # Set up emulators
        remote_cmd_wt(self.machine, cfg.machine.user, cfg.machine.passwd,
                      f"python3 {cfg.machine.workdir}/start_emulator.py --n_emulator {n_device} --base_avd_name {AVD_NAME} "
                      f"--distribution {cfg.sampler}",
                      print_log=self.print_log)
        time.sleep(5)
        # self.queue.put(("update", [machine, 60]))

        # Start client (APP)
        apk_path = os.path.join(cfg.machine.workdir, cfg.fl.apk_path.split("/")[-1])
        cfg_path = os.path.join(cfg.machine.workdir, cfg.fl.cfg_path.split("/")[-1])
        remote_cmd(machine, cfg.machine.user, cfg.machine.passwd,
                   f"python3 {cfg.machine.workdir}/start_client.py --avds_num {n_device} "
                   f"--apk_path {apk_path} "
                   f"--cfg_path {cfg_path} "
                   f"--server_host {pub2pri[cfg.server.host]} "
                   f"--server_port {cfg.server.port} "
                   f"--report_host {pub2pri[machine]}",
                   print_log=self.print_log)
        # self.queue.put(("update", [machine, 100]))


class SetUpEnvProcess(Process):
    def __init__(self, queue, skip_setup_env, cfg, machine):
        super(SetUpEnvProcess, self).__init__()
        self.queue = queue
        self.skip_setup_env = skip_setup_env
        self.cfg = cfg
        self.machine = machine
        self.print_log = DEBUG

    def run(self):
        if not self.skip_setup_env:
            # Step 1.1: Setup java and android environment
            remote_scp(self.machine, self.cfg.machine.user, self.cfg.machine.passwd, "setup_environment.sh", "./",
                       print_log=self.print_log)
            cmd_setup = f"bash ./setup_environment.sh {self.cfg.machine.workdir}"
            remote_cmd(self.machine, self.cfg.machine.user, self.cfg.machine.passwd, cmd_setup,
                       print_log=self.print_log)
        self.queue.put(("update", [machine, 20]))

        # Step 1.2: Copy data, apk, avd, and cfg to remote machine
        # files to pass: data, cfg, apk
        file_list = [self.cfg.fl.cfg_path, self.cfg.fl.apk_path]

        # avd: test if avd exists; transfer avd if not exists
        result_avd = remote_cmd(self.machine, self.cfg.machine.user, self.cfg.machine.passwd,
                                f"ls /{self.cfg.machine.user}/.android/avd/",
                                feedback=True)
        avd_exist = f"{AVD_NAME}.ini" in result_avd
        if not avd_exist:
            file_list += [f"{self.cfg.fl.avd_path}"]
        self.queue.put(("update", [machine, 30]))

        # pass files
        remote_scp(self.machine, self.cfg.machine.user, self.cfg.machine.passwd, file_list, self.cfg.machine.workdir,
                   print_log=self.print_log)
        self.queue.put(("update", [machine, 50]))

        # Uncompress data and avd (if need)
        cmd_unzip_avd = f"unzip -q -o -d /{self.cfg.machine.user}/.android/avd {self.cfg.machine.workdir}/{AVD_NAME}.zip"
        # TODO: rename uncompressed file and delete zip file
        # Uncompress avd
        if not avd_exist:
            remote_cmd(self.machine, self.cfg.machine.user, self.cfg.machine.passwd, cmd_unzip_avd,
                       print_log=self.print_log)
            # Clean the remote file
            cmd_clean_zip = f"rm {self.cfg.machine.workdir}/{AVD_NAME}.zip"
            remote_cmd(self.machine, self.cfg.machine.user, self.cfg.machine.passwd, cmd_clean_zip,
                       print_log=self.print_log)
        self.queue.put(("update", [machine, 100]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=str, help="path of configuration")
    parser.add_argument("--skip_setup_env", type=bool, default=False, help="if skip setting up environment")
    args = parser.parse_args()

    # If exists
    if not os.path.exists(args.cfg):
        print(f"Configuration {args.cfg} not exists")
        exit()

    # Load configuration
    cfg = get_fl_cfg()
    cfg.merge_from_file(args.cfg)

    # For progress bar
    queue = Queue()
    p = Process(target=handle_msg, args=(queue,))
    p.start()

    ################################################################
    #                 STAGE 1: Local Preparation                   #
    ################################################################
    print("\033[1;34m" + "#" * HINT_LENGTH + "\033[0m")
    print("\033[1;34m#" + "STAGE 1: Local Preparation".center(HINT_LENGTH - 2, " ") + "#\033[0m")
    print("\033[1;34m" + "#" * HINT_LENGTH + "\033[0m")

    # Step 1: Check config
    # TODO: for now we suppose all clients use the same config
    print(f"Step 1: Checking config file in {cfg.fl.cfg_path} ".ljust(HINT_LENGTH - 5, "."), end=" ")
    check_file(cfg.fl.cfg_path)
    time.sleep(1)
    print("Done")

    # Step 2: Check apk and avd file
    print(f"Step 2: Checking apk file in {cfg.fl.apk_path} ".ljust(HINT_LENGTH - 5, "."), end=" ")
    check_file(cfg.fl.apk_path)
    print("Done")

    print(f"Step 3: Checking avd file in {cfg.fl.avd_path}".ljust(HINT_LENGTH - 5, "."), end=" ")
    check_file(cfg.fl.avd_path)
    print("Done")

    # Step 4: Check machine list
    print(f"Step 4: Checking remote machines for emulators ".ljust(HINT_LENGTH - 5, "."), end=" ")
    check_file(cfg.machine.host_path)
    print("Done")

    print(f"Step 5: Dividing emulators into different remote machines ".ljust(HINT_LENGTH - 5, "."), end=" ")
    print("Done")
    machines, cpus, memories = list(), list(), list()
    with open(cfg.machine.host_path) as file:
        for line in file:
            ip, cpu, memory = line.replace("\n", "").replace(" ", "").split(",")
            machines.append(ip)
            cpus.append(cpu)
            memories.append(memory)
    n_machines = len(machines)
    assert n_machines * MAX_NUMBER_CLIENT_PER_MACHINE >= cfg.fl.n_client, \
        f"The number of clients {cfg.fl.n_client} is larger than the capacity " \
        f"({n_machines}*{MAX_NUMBER_CLIENT_PER_MACHINE})"
    device_partitions = get_partitions(cfg.fl.n_client, n_machines, machines)

    draw_table(machines, cpus, memories, device_partitions, cfg.sampler)

    MultiProcessBar.set_names(machines[:len(device_partitions)])

    ################################################################
    #                  STAGE 2: Remote Preparation                 #
    ################################################################
    print("\033[1;34m" + "#" * HINT_LENGTH + "\033[0m")
    print("\033[1;34m#" + "STAGE 2: Remote Preparation".center(HINT_LENGTH - 2, " ") + "#\033[0m")
    print("\033[1;34m" + "#" * HINT_LENGTH + "\033[0m")

    # Step 1: setup environment in remote machine
    print(f"Step 1: Set up environment and install packages in remote machine ...")
    MultiProcessBar.init()
    setup_processes = list()
    for i, (machine, n_device) in enumerate(zip(machines, device_partitions)):
        p = SetUpEnvProcess(queue, args.skip_setup_env, cfg, machine)
        setup_processes.append(p)
        p.start()
    for p in setup_processes:
        p.join()
    MultiProcessBar.close()
    print("Done")

    ################################################################
    #                  STAGE 3: Setup Emulator                     #
    ################################################################
    print("\033[1;34m" + "#" * HINT_LENGTH + "\033[0m")
    print("\033[1;34m#" + "STAGE 3: Setup Emulators".center(HINT_LENGTH - 2, " ") + "#\033[0m")
    print("\033[1;34m" + "#" * HINT_LENGTH + "\033[0m")

    # Step 1: kill all existing emulators
    print("Step 1: Kill all existing emulators in remote machine ...")
    MultiProcessBar.init()
    kill_processes = list()
    for i, (machine, n_device) in enumerate(zip(machines, device_partitions)):
        p = KillEmuProcess(queue, cfg, machine)
        kill_processes.append(p)
        p.start()
    for p in kill_processes:
        p.join()
    MultiProcessBar.close()
    print("Done")

    print("\nStep 2: Start android emulators and FS-REAL client in remote machine ...")
    # Step 2: start the emulator
    # MultiProcessBar.init()
    start_processes = list()
    for i, (machine, n_device) in enumerate(zip(machines, device_partitions)):
        p = StartProcess(cfg, queue, machine, n_device)
        start_processes.append(p)
        p.start()
    for p in start_processes:
        p.join()
    # MultiProcessBar.close()
    print("Done")

    queue.put(("end", None))
    p.join()
