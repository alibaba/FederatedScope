# -*- coding:UTF-8 -*-
import numpy as np

import subprocess
import threading
import argparse
import random
import time
import os

from tqdm import tqdm

OPTIONS_POOL = {
    "-cores": [1, 2, 3, 4],
    "-memory": [2048],  # in MB

    # latency is in ms
    # gsm - GSM/CSD (min 150，max 550)
    # edge - EDGE/EGPRS (min 80，max 400)
    # umts - UMTS/3G (min 35，max 200)
    # lte - LTE (min 0，max 0)
    "-netdelay": ["umts", "edge", "lte"],

    # kbps, max upload and download speeds
    # gsm - GSM/CSD (upload: 14.4, download: 14.4)
    # edge - EDGE/EGPRS (upload: 473.6, download: 473.6)
    # umts - UMTS/3G (upload: 384.0, download: 384.0)
    # lte - LTE (upload: 58,000, download: 173,000)
    "-netspeed": ["lte", "evdo", "340000:1024000"]
}

MAX_WAIT_SECOND_OFFLINE = 30
MAX_WAIT_SECOND_FAIL = 15
CMD_START_EMULATOR = dict()
MAX_RESTART_TIMES = 10
TIME_INTERVAL = 5


def sample_beta_binomial(a, b, n, size=None):
    p = np.random.beta(a, b, size=size)
    r = np.random.binomial(n, p)

    return r


class EmulatorSampler:
    def __init__(self, prior="fix"):
        self.prior = prior

    def sample(self, size):
        if self.prior.lower() == "fix":
            return ["-cores 1 -memory 2048 -netdelay lte -netspeed lte" for _ in range(size)]
        elif self.prior.lower() == "fix_good":
            return ["-cores 2 -memory 2048 -netdelay lte -netspeed 340000:1024000" for _ in range(size)]
        elif self.prior.lower() == "fix_bad":
            return ["-cores 1 -memory 2048 -netdelay umts -netspeed 30000:50000" for _ in range(size)]
        elif self.prior.lower() == "high":
            return ["-cores 4 -memory 2048 -netdelay lte -netspeed lte" for _ in range(size)]
        elif self.prior.lower() == "uniform":
            # sample options
            choice_options = dict()
            for option, pool in OPTIONS_POOL.items():
                choice_options[option] = np.random.choice(pool, size=size).tolist()
            # generate cmd for options
            emulators = list()
            for i in range(size):
                cmd_options = " ".join([f"{key} {values[i]}" for key, values in choice_options.items()])
                emulators.append(cmd_options)
            return emulators
        elif self.prior.lower().startswith("device"):
            cfg_id = int(self.prior.lower().replace("device", ""))
            cpu_id = cfg_id // 9
            speed_id = cfg_id % 9 // 3
            delay_id = cfg_id % 9 % 3
            cmd_options = f"-cores {OPTIONS_POOL['-cores'][cpu_id]} -memory 2048 -netdelay {OPTIONS_POOL['-netdelay'][delay_id]} -netspeed {OPTIONS_POOL['-netspeed'][speed_id]}"
            return [cmd_options for _ in range(size)]
        else:
            if self.prior.lower() == "a10b10":
                alpha, beta = 10, 10
            elif self.prior.lower() == "a10b2":
                alpha, beta = 10, 2
            elif self.prior.lower() == "a2b10":
                alpha, beta = 2, 10
            elif self.prior.lower() == "a2b2":
                alpha, beta = 0.2, 0.2
            else:
                raise ValueError(f"{self.prior.lower()}")

            choice_options = dict()
            for option, pool in OPTIONS_POOL.items():
                choice_idx = sample_beta_binomial(alpha, beta, len(pool)-1, size)
                choice_options[option] = [pool[_] for _ in choice_idx]

            emulators = list()
            for i in range(size):
                cmd_options = " ".join([f"{key} {values[i]}" for key, values in choice_options.items()])
                emulators.append(cmd_options)
            return emulators


def get_cur_status():
    output = subprocess.getoutput("adb devices")
    # Delete "\n" in the last
    if output.endswith("\n"):
        output = output[:-1]
    status = output.split("\n")[1:]
    online_list, offline_list = list(), list()
    for _ in status:
        name, state = _.split("\t")
        if state == "offline":
            offline_list.append(name)
        elif state == "device":
            online_list.append(name)
        else:
            raise ValueError()

    return len(status), online_list, offline_list


def check_emulators(n_emulator, distribution):
    start_time = time.time()
    pbar = tqdm(total=n_emulator, bar_format="{percentage:3.0f}%|{bar}|{n}/{total}[{elapsed}]")
    restart_times = 0

    while True:
        time.sleep(TIME_INTERVAL)
        n_setup, online_list, offline_list = get_cur_status()
        if len(online_list) > 150:
            raise ValueError

        if len(online_list) - pbar.n == 0:
            pbar.refresh()
        else:
            pbar.update(len(online_list) - pbar.n)

        # Devices that are set up successfully, including online and offline
        all_list = online_list + offline_list
        # Failed emulators
        failed_emulator = list(set(CMD_START_EMULATOR.keys()) - set(all_list))
        if len(failed_emulator) == 0:
            # all emulators are set up
            if len(online_list) == n_emulator:
                break
            elif time.time() - start_time <= MAX_WAIT_SECOND_OFFLINE:
                continue
            else:
                # Restart the offline emulators
                pbar.write(f"Timeout! Re-start offline emulators {offline_list}")
                for avd_name in offline_list:
                    # Kill the emulator
                    cmd = f"adb -s ${avd_name} emu kill"
                    os.system(cmd)
                    # Restart
                    os.system(CMD_START_EMULATOR[avd_name])
                    # refresh time
                    start_time = time.time()
                restart_times += 1
        elif time.time() - start_time <= MAX_WAIT_SECOND_FAIL:
            continue
        else:
            pbar.write(f"Timeout! Re-start failed emulators {failed_emulator}")
            # Re-start the failed emulators
            new_options_list = resample_configs(len(failed_emulator), distribution)
            for avd_name, new_options in zip(failed_emulator, new_options_list):
                emulator_port = avd_name.split("-")[-1]
                emulator_id = int((port - 5554) / 2)

                new_cmd = f"emulator -avd {args.base_avd_name} -id {emulator_id} " \
                          f"{new_options} -no-window -no-audio -read-only -writable-system " \
                          f"-port {emulator_port}"
                new_cmd = f"nohup {new_cmd} >/dev/null 2>&1 & "
                # resample configs
                os.system(new_cmd)
                CMD_START_EMULATOR[avd_name] = new_cmd

            restart_times += 1
            # refresh time
            start_time = time.time()

        if restart_times > MAX_RESTART_TIMES:
            print(
                f"Re-start emulator for {restart_times} times, emulators {failed_emulator + offline_list} still failed")
            print(f"The commands for failed emulators are:")
            for avd_name in failed_emulator + offline_list:
                print(f"--{avd_name}:")
                print(f"\t{CMD_START_EMULATOR[avd_name]}")
            print(f"All emulators will be killed.")
            break

    pbar.close()


def resample_configs(n_failed, distribution):
    resampler = EmulatorSampler(distribution)
    new_options = resampler.sample(n_failed)
    return new_options


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_emulator", help="the number of emulators", type=int, required=True)
    parser.add_argument("--base_avd_name", help="the name of basic avd", type=str, required=True)
    parser.add_argument("--distribution", type=str, required=True)
    args = parser.parse_args()

    sampler = EmulatorSampler(args.distribution)

    # Step 1: Restart adb to enable port access
    print(f"Step 1: Restart adb server ...")
    os.system("adb kill-server")
    os.system("nohup adb -a nodaemon server start >/dev/null 2>&1 &")
    print("Done")

    time.sleep(3)

    # Obtain emulator options for each machine
    options_list = sampler.sample(args.n_emulator)
    print("Start to setup emulators ...")
    for i, options in enumerate(options_list):
        port = 2 * i + 5554
        # TODO: path
        cmd = f"emulator -avd {args.base_avd_name} -id {i} " \
              f"{options} -no-window -no-audio -read-only -writable-system " \
              f"-port {port}"
        cmd = f"nohup {cmd} >/dev/null 2>&1 & "
        os.system(cmd)
        CMD_START_EMULATOR[f"emulator-{port}"] = cmd

        time.sleep(0.5)

    # Check if enough emulators are started
    print("\nChecking the emulators ...")
    check_emulators(args.n_emulator, args.distribution)
    # thread = threading.Thread(target=check_emulators, args=(args.n_emulator, ))
    # thread.start()
    #
    # thread.join()

    # Finish
    # print(CMD_START_EMULATOR)
    print("Finished")
