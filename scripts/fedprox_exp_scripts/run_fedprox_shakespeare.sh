set -e

cudaid=$1

if [ ! -d "out_fedprox_shakespeare" ];then
  mkdir out_fedprox_shakespeare
fi

echo "fedprox Shakespeare starts..."

mus=(0.01 0.1 1. 10 100)

for ((im=0; im<${#mus[@]}; im++ ))
do
  python federatedscope/main.py --cfg federatedscope/nlp/baseline/fedavg_lstm_on_shakespeare.yaml device ${cudaid} \
  data.root /mnt/gaodawei.gdw/data/ \
  fedprox.use True \
  fedprox.mu ${mus[$im]} \
  >>out_fedprox_shakespeare/nothing.out \
  2>>out_fedprox_shakespeare/mu_${mus[$im]}.log
done

for ((im=0; im<${#mus[@]}; im++ ))
do
  python federatedscope/../scripts/fedprox_exp_scripts/parse_mf_results.py --input out_fedprox_shakespeare/mu_${mus[$im]}.log \
  --round 1000
done

echo "Ends."

