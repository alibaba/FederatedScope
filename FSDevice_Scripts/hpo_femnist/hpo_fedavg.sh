#!/bin/bash

set -e

server_host=${1}
server_port=${2}
server_username=${3}
server_passwd=${4}
client_username=${5}
client_passwd=${6}
# e.g. /root/FS-Device/FS-Device-MNN
server_dir=${7}
# e.g. /root/FS-Device/FSDevice_Scripts
script_dir=${8}

cmd_python=$(which python)
cmd_mnn=$(which mnnconvert)

if [ ! -d "${server_dir}" ]; then
  echo "${server_dir} doesn't exist"
  exit
fi

if [ ! -d "${script_dir}" ]; then
  echo "${script_dir} doesn't exist"
  exit
fi

if [ ! -d "${script_dir}/${config_dir}" ]; then
  echo "${script_dir}/${config_dir} doesn't exist"
  exit
fi

# Scale
RECEIVE=(90 179 359 718)
SAMPLE=(135 269 539 1078)
EXECUTOR=(145 279 549 1088)
SEEN=(450 899 1798 3596)

# Parameters
MOM=(0.)
WD=(0.)
LR=(0.1)
LUS=(1)
SEED=(0)
DEVICE=(fix uniform a10b10 a10b2 a2b2)


DumpConfig(){
  echo "workdir: \"\"
sampler: \"${1}\"
server:
  host: \"${server_host}\"
  port: ${server_port}
  workdir: \"/root/FS-Device\"
  user: \"${server_username}\"
  passwd: \"${server_passwd}\"
machine:
  host_path: \"${config_dir}/machines.txt\"
  workdir: \"/root/FS-Device\"
  user: \"${client_username}\"
  passwd: \"${client_passwd}\"
fl:
  n_client: ${2}
  cfg_path: \"${config_dir}/client_cfg.yaml\"
  apk_path: \"material/apk/fsandroid/debug/app-debug.apk\"
  avd_path: \"material/avd/test_x86.zip\"" > ${script_dir}/${config_dir}/config.yaml
}

DumpClientConfig(){
  echo "data:
  batch_size: 16
  root: \"/data/user/0/com.example.fsandroid/files/\"
  type: \"femnist\"
distribute:
  device_port: 50078
  report_host: \"\"
  report_port: 50078
  server_host: \"\"
  server_port: 50051
  grpc_compression: \"\"
model:
  type: \"ConvNet2\"
optimizer:
  lr: ${1}
  momentum: ${2}
  weight_decay: ${3}
task:
  criterion: \"CrossEntropyLoss\"
  type: \"classification\"
  category: 62
finetune:
  use: false
  local_update_steps: 0
  lr: 0.
  weight_decay: 0.
  momentum: 0.
train:
  auto_start: true
  local_update_steps: ${4}" > ${script_dir}/${config_dir}/fedavg_client_cfg.yaml
}


for i in "${!SAMPLE[@]}"; do
  sample_num=${SAMPLE[i]}
  seen_num=${SEEN[i]}
  received_num=${RECEIVE[i]}
  executor_num=${EXECUTOR[i]}

  for mom in ${MOM[*]}
  do
    for wd in ${WD[*]}
    do
      for seed in ${SEED[*]}
      do
        for lr in ${LR[*]}
        do
          for lus in ${LUS[*]}
          do
            for device in ${DEVICE[*]}
            do
              # Exp name
              expname="FEMNIST_FEDAVG_DEVICE-${device}_RECEIVE-${received_num}_SAMPLE-${sample_num}_EXECUTOR-${executor_num}_SEEN-${seen_num}_LR-${lr}_LUS-${lus}_MOM-${mom}_WD-${wd}_SEED-${seed}"
              # Write to the same log
              logfile="${script_dir}/exp/federated/femnist/log/${expname}.log"

              echo "Device: ${device}, Received_num: ${received_num}, Sample_num: ${sample_num}, Executor_num: ${executor_num}, Seen_num: ${seen_num}, Lr: ${lr}, Lus: ${lus}, Mom: ${mom}, Wd: ${wd}, Seed: ${seed}" | tee -a "${logfile}"
              # Prepare device scripts
              DumpClientConfig "${lr}" "${mom}" "${wd}" "${lus}"
              # Modify emulator number
              n_setup_executor=`expr ${executor_num} + 5`
              echo "Start ${n_setup_executor} android emulators"
              DumpConfig "${device}" "${n_setup_executor}"

              # Setup server in a process
              cd "${server_dir}"
              ${cmd_python} federatedscope/main.py --cfg scripts/device_scripts/server_femnist.yaml \
                                seed ${seed} \
                                distribute.server_port ${server_port} \
                                federate.sample_client_num ${sample_num} \
                                federate.seen_client_num ${seen_num} \
                                federate.min_received_num ${received_num} \
                                federate.executor_num ${executor_num} \
                                expname ${expname} \
                                mnn.cmd_convert ${cmd_mnn} &

              # Setup emulator in a process
              cd "${script_dir}"
              ${cmd_python} 02_setup_device.py --cfg ${config_dir}/config.yaml --skip_setup_env True &

              # Wait until they are finished
              wait

              ${cmd_python} 03_stop_emulator.py --cfg ${config_dir}/config.yaml

              echo "==================================================================================================="
            done
          done
        done
      done
    done
  done
done
