import argparse
import multiprocessing
import os

import subprocess
import time
import yaml
from tqdm import tqdm

"""
An example of config file:
    data:
      batch_size:         32
      root:               /data/user/0/com.example.fsandroid/files/
      type:               MNIST
    distribute:
      device_port:        50078
      report_port:        50078
      report_host:        0.0.0.0
      server_host:        8.210.21.85
      server_port:        50051
    model:
      type:               LeNet
    optimizer:
      lr:                 0.1
      momentum:           0.0
      weight_decay:       0.0
    task:
      criterion:          CrossEntropyLoss
      type:               classification
    train:
      auto_start:         true
      local_update_steps: 1
"""

DEBUG = False
NO_DEBUG_SUFFIX = " >/dev/null 2>&1"
DEVICE_BASE_PORT = 6000


def local_cmd(cmd):
    if DEBUG:
        os.system(cmd)
    else:
        os.system(cmd + NO_DEBUG_SUFFIX)


def start_app(args, cfg, c_i, device):
    report_port = DEVICE_BASE_PORT + c_i
    device_port = DEVICE_BASE_PORT + c_i

    # Stop the app if it is running
    cmd = f"adb -s {device} shell am force-stop com.example.fsandroid"
    local_cmd(cmd)

    cmd = f"adb -s {device} shell run-as com.example.fsandroid mkdir files"
    local_cmd(cmd)

    # TODO: Check if datasets are moved successfully

    # cfg file generation and perturbation(todo)
    # change machine port
    cfg["distribute"]["device_port"] = device_port
    cfg["distribute"]["report_host"] = args.report_host
    cfg["distribute"]["report_port"] = report_port
    cfg["distribute"]["server_host"] = args.server_host
    cfg["distribute"]["server_port"] = args.server_port

    # move cfg file into the internal directory of app
    file_config = f"{report_port}_config.yaml"
    with open(file_config, 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)
    cmd = f"adb -s {device} push {file_config} /data/local/tmp"
    local_cmd(cmd)

    cmd = f"adb -s {device} shell run-as com.example.fsandroid rm files/config.yaml"
    local_cmd(cmd)

    cmd = f"adb -s {device} shell run-as com.example.fsandroid cp /data/local/tmp/{file_config} files/config.yaml"
    local_cmd(cmd)

    # port mapping for the emulated devices
    cmd = f"adb -s {device} forward tcp:{report_port} tcp:{device_port}"
    local_cmd(cmd)

    # start the app
    cmd = f"adb -s {device} shell am start -n com.example.fsandroid/.MainActivity"
    local_cmd(cmd)

    os.remove(file_config)
    print(f"Finish starting app in the {c_i}-th device")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--avds_num",
                        required=True,
                        help="the number of emulated devices",
                        type=int)
    parser.add_argument("--apk_path",
                        required=True,
                        help="the path of apk to be installed",
                        type=str)
    parser.add_argument("--cfg_path",
                        required=True,
                        help="the path of basic config",
                        type=str)
    parser.add_argument("--server_host",
                        required=True,
                        type=str)
    parser.add_argument("--server_port",
                        required=True,
                        type=int)
    parser.add_argument("--report_host",
                        required=True,
                        type=str)
    args = parser.parse_args()

    # Step 0: check apk, cfg, data
    for path in [args.apk_path, args.cfg_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} doesn't exists")

    # after installing, get the running devices
    devices = set()
    process = subprocess.Popen("adb devices".split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    for line in output.decode("utf-8").strip().split("\n"):
        res = line.split("\t")
        # e.g., ['List of devices attached', 'emulator-5556\tdevice',
        # 'emulator-5558\tdevice']
        if len(res) == 2 and res[1] == "device":
            devices.add(res[0])
            print(f"GET RUNNING DEVICE: {res[0]}")

    print("======" * 5)
    if len(devices) != args.avds_num:
        print(f"Failed to starting some devices. "
              f"Expect {args.avds_num} devices, but only found {len(devices)}")

    # Step 1: Install app
    print(f"Step 1: Start to install apk {args.apk_path} ...")
    for device_name in tqdm(devices):
        cmd = f"adb -s {device_name} uninstall com.example.fsandroid"
        local_cmd(cmd)

        cmd = f"adb -s {device_name} install -t {args.apk_path}"
        local_cmd(cmd)

    # Step 2: Check if installed successfully
    install_success_devices = set()
    # check the apk install status and try to re-install at most three times
    for install_time in range(3):
        print(f"[{install_time}/3] Wait 2 seconds, and then check the status ")
        time.sleep(2)
        if len(install_success_devices) == len(devices):
            break
        for device_name in devices:
            if device_name not in install_success_devices:
                process = subprocess.Popen(
                    f"adb -s {device_name} shell pm list packages |"
                    " grep com.example.fsandroid".split(),
                    stdout=subprocess.PIPE)
                output, error = process.communicate()
                if output != "":
                    install_success_devices.add(device_name)
                else:
                    # try to re-install
                    cmd = f"adb -s {device_name} install -t {args.apk_path}"
                    local_cmd(cmd)

    install_fail_devices = devices - install_success_devices
    print(f"Totally {len(install_fail_devices)} devices "
          f"failed to install the apk")
    for device in install_fail_devices:
        print(f"Device {device} failed to install the apk")

    # TODO: re-install apk

    # Moving data, cfg
    print(f"Step 3: Start to moving dataset and cfg to devices ...")
    # Prepare config
    cfg = yaml.safe_load(open(args.cfg_path))
    # Prepare adb port
    local_cmd("adb forward --remove-all")

    print(f"Dispatch dataset and cfg, and start app ...")

    pool = multiprocessing.Pool(processes=min(len(install_success_devices), 50))
    for c_i, device in enumerate(sorted(install_success_devices)):
        pool.apply_async(start_app, (args, cfg, c_i, device))

    pool.close()
    pool.join()
    print("Finished!")
