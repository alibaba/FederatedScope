set -e

cudaid=$1

if [ ! -d "out_vfl1m" ];then
  mkdir out_vfl1m
fi

echo "VFL starts..."

lrs=(1.2 1. 0.8 0.6 0.1)
steps=(1 5 10 20 50)
batchs=(8 16 32 64)


for ((ib=0; ib<${#batchs[@]}; ib++ ))
do
  for ((is=0; is<${#steps[@]}; is++ ))
  do
    for ((il=0; il<${#lrs[@]}; il++ ))
    do
      round=$[1000/${steps[$is]}]
      python federatedscope/main.py --cfg federatedscope/mf/baseline/vfl_fedavg_standalone_on_movielens1m.yaml device ${cudaid} \
      data.root /mnt/gaodawei.gdw/data/ \
      sgdmf.use False \
      train.optimizer.lr ${lrs[$il]} \
      train.local_update_steps ${steps[$is]} \
      federate.total_round_num ${round} \
      data.batch_size ${batchs[$ib]}  \
      >>out_vfl1m/nothing.out \
      2>>out_vfl1m/batch_${batchs[$ib]}_round_${round}_steps_${steps[$is]}_lr_${lrs[$il]}.log
    done
  done
done

for ((ib=0; ib<${#batchs[@]}; ib++ ))
do
  for ((is=0; is<${#steps[@]}; is++ ))
  do
    round=$[1000/${steps[$is]}]
    for ((il=0; il<${#lrs[@]}; il++ ))
    do
      python federatedscope/../scripts/mf_exp_scripts/parse_mf_results.py --input out_vfl1m/batch_${batchs[$ib]}_round_${round}_steps_${steps[$is]}_lr_${lrs[$il]}.log \
      --round ${round}
    done
  done
done

echo "Ends."

