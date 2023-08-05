# FS-REAL

## Introduction

FS-REAL is an efficient and scalable prototyping system for real-world cross-device federated learning. More details can be found in our [paper](link_to_be_added).
This repository includes the implementation of its FL server, the FL clients runtime on android devices, the device simulation tool and running scripts. 

- `FS-Device-MNN`:  Codes for FL server in FS-REAL.
- `FsOnAndroid`:   Codes for FL clients in FS-REAL.   
- `FSDevice_Scripts`:   Codes for running scripts and simulation tools.

*The full codes are coming soon!*

## Guidance

### Preparation
To start a distributed FL course with FS-REAL, you need to prepare
- **SERVER MACHINE**: One machine (could be a cloud server or personal PC) that can be accessed via public or LAN address, which is used to coordinate the FL devices.
- **CLIENT MACHINES**: Several machines that can be accessed via public or LAN address, which are used to run android emulators. 
- APK file: An APK file to install the APP for clients in emulators, which is built from `FsOnAndroid`
- AVD file: Android avd file that characterizes the hardware configuration of device
  - We have preset an avd file in ```material/avd```. You don't need to modify it if your **CLIENT MACHINES** are all on x86 architecture.

### Set-up Server
All the following steps are executed on **SERVER MACHINE**: 
- Follow [README.md of FederatedScope](https://github.com/alibaba/FederatedScope/blob/master/README.md) to install the required packages in **SERVER MACHINE**
- Copy the code for FS-REAL server ```FS-Device-MNN``` to your **SERVER MACHINE**
- Start to run an FS-REAL server as follows, and you can change the detailed configurations in the yaml.
  - An example yaml is in ```scripts/device_scripts/server_femnist.yaml```
```
cd FS-Device-MNN
python federatedscope/main.py --cfg ${PATH_OF_YOUR_YAML}
```

If the installation and configuration work well, you will see the log message similar to the one below, indicating that the Server is listening for clients to join FL:

![Start FSReal server](https://img.alicdn.com/imgextra/i3/O1CN01lJHiah1trtmXYvBil_!!6000000005956-0-tps-5114-298.jpg)



### Set-up Client

Firstly,  users need to set up the **CLIENT MACHINES** by preparing running configurations, including
- config.yaml: config file for the running scripts, including username and password for server and client machines, working dir and paths of avd and APK file

- client_cfg.yaml: config file for android client

- machines.txt: list the public/LAN address for all **CLIENT MACHINES** (one address in each line)
-  (We provide examples in ```FSDevice_Scripts/exp/femnist```.)

Secondly, run the following scripts to set up emulators and start FS-REAL clients `02_setup_device.py`, which will install the required packages automatically, including
- Necessary APT packages and python packages
- JDK 8
- Android sdk and sdk tools to conduct the simulation:
  - sdkmanager
  - emulator
  - platform-tools
  - platforms;android-30
  - system-images;android-30;google_apis;x86_64

```
python 02_setup_device.py --cfg ${PATH_OF_YOUR_CLIENTS_CONFIG}
``` 

And you will see that the clients connect to the server as follows if everything works well:

![Clients join in federated training](https://img.alicdn.com/imgextra/i4/O1CN01CORMSG28vIxrG4IGs_!!6000000007994-0-tps-2466-1738.jpg)

### Stop All Emulators

To stop the emulators, run script as follows
```
python 03_stop_emulator.py --cfg ${PATH_OF_YOUR_CONFIG}
```

And you will see

![03_stop_emulator.py](https://img.alicdn.com/imgextra/i3/O1CN01G3tR2s21eAq5gOz4r_!!6000000007009-0-tps-2870-919.jpg)

### HPO

We provide some example scripts for HPO in the dir ```hpo_femnist```. Please refer to the scripts for more details.

### Note
- If you have changed the code in FsOnAndroid, you need to rebuild the APK file in debug mode.

