set -e

cudaid=$1

if [ ! -d "out_fedopt_shakespeare" ];then
  mkdir out_fedopt_shakespeare
fi

echo "FedOpt Shakespeare starts..."

lrs=(0.1 0.5 0.75 1. 1.25 1.5)

for ((il=0; il<${#lrs[@]}; il++ ))
do
  python flpackage/main.py --cfg flpackage/nlp/baseline/fedavg_lstm_on_shakespeare.yaml device ${cudaid} \
  data.root /mnt/gaodawei.gdw/data/ \
  fedopt.use True \
  federate.method FedOpt \
  fedopt.lr_server ${lrs[$il]} \
  >>out_fedopt_shakespeare/nothing.out \
  2>>out_fedopt_shakespeare/lr_${lrs[$il]}.log
done

for ((il=0; il<${#lrs[@]}; il++ ))
do
  python flpackage/../scripts/fedopt_exp_scripts/parse_mf_results.py --input out_fedopt_shakespeare/lr_${lrs[$il]}.log \
  --round 1000
done

echo "Ends."

