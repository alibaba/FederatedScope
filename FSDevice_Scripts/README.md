# Running Scripts for FS Device

## Introduction

This repository aims to run a large-scale cross-device FL simulation with android emulators, including
setting up FS-REAL server, emulators and device runtimes.

## Quick Start

### Prepare Remote Machines

- To run a simulation, you need to prepare 
  - a remote machine for FS-REAL server
  - at least one remote machine for android emulators

- Note FS-REAL server and android emulators could be located on the same machine. Otherwise, the remote machines should be accessible to each other via LAN or public network.

### Prepare Configuration Files

- Before running a simulation, you need to prepare the following configuration files:
  - `simulation_config.json`: configuration for the simulation, including the number of emulators, the number of devices, the number of clients, etc.
  - `client_config.json`: configuration for each client, including the device type, the device runtime, etc.
  - `server_config.json`: configuration for the FS-REAL server, including the IP address, the port, etc.

### Run Simulation
- Log in the remote machine for FS-REAL server via ssh.
```bash
ssh <username>@<server_ip>
```

- Clone this repository.
```bash
git clone https://github.com/alibaba/FederatedScope
```

- Run simulation as follows
```bash
bash start_simulation.sh ${EXP_NAME} \
                  ${SIMULATION_CONFIG}  \ 
                  ${CLIENT_CONFIG}  \
                  ${SERVER_CONFIG}
```