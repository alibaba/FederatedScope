set -e

alpha=$1
device=$2

echo $alpha
echo $device

CUDA_VISIBLE_DEVICES="${device}" python federatedscope/hpo.py --cfg scripts/wide_valley_exp_scripts/fedentsgd_on_cifar10.yaml hpo.working_folder bo_gp_fedentsgd_${device} outdir bo_gp_fedentsgd_${device} >/dev/null 2>/dev/null
