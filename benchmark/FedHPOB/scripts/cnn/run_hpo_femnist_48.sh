set -e

# wds=(0.0 0.001 0.01 0.1)
# steps=(1 2 3 4)
# batch_sizes=(16 32 64)

cudaid=$1
wd=$2
step=$3
batch_size=$4

cd ../..

dataset=femnist

out_dir=out_${dataset}

echo "HPO starts..."

lrs=(0.01 0.01668 0.02783 0.04642 0.07743 0.12915 0.21544 0.35938 0.59948 1.0)
dps=(0.0 0.5)
sample_rates=(0.2 0.4 0.6 0.8 1.0)

for ((l = 0; l < ${#lrs[@]}; l++)); do
  for ((d = 0; d < ${#dps[@]}; d++)); do
    for ((s = 0; s < ${#sample_rates[@]}; s++)); do
      for k in {1..3}; do
        python federatedscope/main.py --cfg fedhpo/cnn/${dataset}.yaml device $cudaid optimizer.lr ${lrs[$l]} optimizer.weight_decay ${wd} model.dropout ${dps[$d]} federate.local_update_steps ${step} data.batch_size ${batch_size} federate.sample_client_rate ${sample_rates[$s]} seed $k outdir ${out_dir}_${sample_rates[$s]} expname lr${lrs[$l]}_wd${wd}_dropout${dps[$d]}_step${step}_batch${batch_size}_seed${k} >/dev/null 2>&1
      done
    done
  done
done

echo "HPO ends."
