set -e

algo=$1
alpha=$2

for (( i=0; i<5; i++ ))
do
  CUDA_VISIBLE_DEVICES="${i}" python federatedscope/main.py --cfg scripts/wide_valley_exp_scripts/${algo}_on_cifar10.yaml seed $i data.splitter_args "[{'alpha': ${alpha}}]" expname ${algo}_${alpha}_${i} &
done
