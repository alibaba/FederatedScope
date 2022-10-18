# https://huggingface.co/google/bert_uncased_L-2_H-128_A-2
set -e

cudaid=$1
sample_rate=$2
dataset=$3
k=$4

cd ../..

out_dir=out_${dataset}

echo "HPO starts..."

lrs=(0.01 0.01668 0.02783 0.04642 0.07743 0.12915 0.21544 0.35938 0.59948 1.0)
wds=(0.0 0.001 0.01 0.1)
dps=(0.0 0.5)
steps=(1)
batch_sizes=(8 16 32 64 128)

lrs_server=(0.1 0.5 1.0)
momentums_server=(0.0 0.9)

for ((l = 0; l < ${#lrs[@]}; l++)); do
  for ((w = 0; w < ${#wds[@]}; w++)); do
    for ((d = 0; d < ${#dps[@]}; d++)); do
      for ((s = 0; s < ${#steps[@]}; s++)); do
        for ((b = 0; b < ${#batch_sizes[@]}; b++)); do
          for ((sl = 0; sl < ${#lrs_server[@]}; sl++)); do
            for ((ms = 0; ms < ${#momentums_server[@]}; ms++)); do
              python main.py --cfg fedhpo/glue/${dataset}.yaml fedopt.use True federate.method FedOpt fedopt.lr_server ${lrs_server[$sl]} fedopt.momentum_server ${momentums_server[$ms]} device $cudaid optimizer.lr ${lrs[$l]} optimizer.weight_decay ${wds[$w]} model.dropout ${dps[$d]} federate.local_update_steps ${steps[$s]} data.batch_size ${batch_sizes[$b]} federate.sample_client_rate $sample_rate seed $k outdir out_fedopt/${out_dir}_${sample_rate} federate.share_local_model False federate.online_aggr False expname lr${lrs[$l]}_wd${wds[$w]}_dropout${dps[$d]}_step${steps[$s]}_batch${batch_sizes[$b]}_lrserver${lrs_server[$sl]}_momentumsserver${momentums_server[$ms]}_seed${k} >/dev/null 2>&1
            done
          done
        done
      done
    done
  done
done

echo "HPO ends."