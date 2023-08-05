#!/bin/bash

set -e

exp_name=${1}
simulation_config=${2}
client_config=${3}
server_config=${4}

# path for code
# e.g. /root/FS-Device/FS-Device-MNN
server_dir=${5}
# e.g. /root/FS-Device/FSDevice_Scripts
script_dir=${6}

cmd_python=$(which python)

# Exp name
echo "======================================================================================"
echo -e "\033[;37m TASK (${exp_name}): \033[0m"
echo "  SIMULATION CONFIG:  ${simulation_config}"
echo "  CLIENT CONFIG:      ${client_config}"
echo "  SERVER CONFIG:      ${server_config}"
echo "======================================================================================"

echo -e "\033[;37m Setup FS-REAL server: \033[0m"
# Setup server in a process
cd "${server_dir}"
${cmd_python} federatedscope/main.py --cfg "${server_config}" \
                  expname "${exp_name}" \
                  mnn.cmd_convert $(which mnnconvert) &

echo "======================================================================================"
echo -e "\033[;37m Setup FS-REAL clients in bulk: \033[0m"
# Setup emulator in a process
cd "${script_dir}"
${cmd_python} 02_setup_device.py --cfg ${simulation_config} --skip_setup_env True &

# Wait until they are finished
wait

echo "======================================================================================"
exit

echo "Stop all emulators ..."
${cmd_python} 03_stop_emulator.py --cfg ${simulation_config} >/dev/null 2>&1
echo "Done"


