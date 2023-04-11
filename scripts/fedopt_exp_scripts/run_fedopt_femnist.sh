set -e

cudaid=$1

if [ ! -d "out_fedopt_femnist" ];then
  mkdir out_fedopt_femnist
fi

echo "VFL starts..."

lrs=(0.1 0.5 0.75 1. 1.25 1.5)

for ((il=0; il<${#lrs[@]}; il++ ))
do
  python federatedscope/main.py --cfg federatedscope/cv/baseline/fedavg_convnet2_on_femnist.yaml device ${cudaid} \
  data.root /mnt/gaodawei.gdw/data/ \
  fedopt.use True \
  federate.method FedOpt \
  fedopt.optimizer.lr ${lrs[$il]} \
  >>out_fedopt_femnist/nothing.out \
  2>>out_fedopt_femnist/lr_${lrs[$il]}.log
done

for ((il=0; il<${#lrs[@]}; il++ ))
do
  python federatedscope/../scripts/fedopt_exp_scripts/parse_mf_results.py --input out_fedopt_femnist/lr_${lrs[$il]}.log \
  --round 300
done

echo "Ends."

