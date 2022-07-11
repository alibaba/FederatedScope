set -e

cudaid=$1

if [ ! -d "out_vflsgdmf1m" ];then
  mkdir out_vflsgdmf1m
fi

echo "VFL starts..."

epsilons=(0.5 2. 4.)
deltas=(0.25 0.5 0.75)
lrs=(1.5 1. 0.5 0.1)
steps=(1 5 10 20 50)
batchs=(8 16 32 64)


for ((ib=0; ib<${#batchs[@]}; ib++ ))
do
  for ((is=0; is<${#steps[@]}; is++ ))
  do
    round=$[1000/${steps[$is]}]
    for ((ie=0; ie<${#epsilons[@]}; ie++ ))
    do
      for ((id=0; id<${#deltas[@]}; id++ ))
      do
        for ((il=0; il<${#lrs[@]}; il++ ))
        do
          python federatedscope/main.py --cfg federatedscope/mf/baseline/vfl-sgdmf_fedavg_standalone_on_movielens1m.yaml device ${cudaid} \
          data.root /mnt/gaodawei.gdw/data/ \
          sgdmf.use True \
          sgdmf.epsilon ${epsilons[$ie]} \
          sgdmf.delta ${deltas[$id]} \
          train.optimizer.lr ${lrs[$il]} \
          train.local_update_steps ${steps[$is]} \
          federate.total_round_num ${round} \
          data.batch_size ${batchs[$ib]}  \
          >>out_vflsgdmf1m/temp.out \
          2>>out_vflsgdmf1m/batch_${batchs[$ib]}_round_${round}_steps_${steps[$is]}_eps_${epsilons[$ie]}_delta_${deltas[$id]}_lr_${lrs[$il]}.log
        done
      done
    done
  done
done

for ((ib=0; ib<${#batchs[@]}; ib++ ))
do
  for ((is=0; is<${#steps[@]}; is++ ))
  do
    round=$[1000/${steps[$is]}]
    for ((ie=0; ie<${#epsilons[@]}; ie++ ))
    do
      for ((id=0; id<${#deltas[@]}; id++ ))
      do
        for ((il=0; il<${#lrs[@]}; il++ ))
        do
          python federatedscope/../scripts/mf_exp_scripts/parse_mf_results.py \
          --input out_vflsgdmf1m/batch_${batchs[$ib]}_round_${round}_steps_${steps[$is]}_eps_${epsilons[$ie]}_delta_${deltas[$id]}_lr_${lrs[$il]}.log \
          --round ${round}
        done
      done
    done
  done
done

echo "Ends."

