set -e

cudaid=$1
dataset=$2

cd ../..

out_dir=out_${dataset}

if [ ! -d $out_dir ]; then
  mkdir $out_dir
fi

if [[ $dataset == '10101' ]]; then
  out_channels=2
elif [[ $dataset == '53' ]]; then
  out_channels=4
elif [[ $dataset == '146818' ]]; then
  out_channels=2
elif [[ $dataset == '146821' ]]; then
  out_channels=4
elif [[ $dataset == '9952' ]]; then
  out_channels=2
elif [[ $dataset == '146822' ]]; then
  out_channels=7
elif [[ $dataset == '31' ]]; then
  out_channels=2
elif [[ $dataset == '3917' ]]; then
  out_channels=2
elif [[ $dataset == '168912' ]]; then
  out_channels=2
elif [[ $dataset == '3' ]]; then
  out_channels=2
elif [[ $dataset == '167119' ]]; then
  out_channels=3
elif [[ $dataset == '12' ]]; then
  out_channels=10
elif [[ $dataset == '146212' ]]; then
  out_channels=7
elif [[ $dataset == '168911' ]]; then
  out_channels=2
elif [[ $dataset == '9981' ]]; then
  out_channels=9
elif [[ $dataset == '167120' ]]; then
  out_channels=2
elif [[ $dataset == '14965' ]]; then
  out_channels=2
elif [[ $dataset == '146606' ]]; then
  out_channels=2
elif [[ $dataset == '7592' ]]; then
  out_channels=2
elif [[ $dataset == '9977' ]]; then
  out_channels=2
else
  out_channels=2
fi

echo "HPO starts..."

sample_rates=(0.2 0.4 0.6 0.8 1.0)
lrs=(0.00001 0.0001 0.001 0.01 0.1 1.0)
wds=(0.0 0.001 0.01 0.1)
steps=(1)
batch_sizes=(4 8 16 32 64 128 256)

lrs_server=(0.1 0.5 1.0)
momentums_server=(0.0 0.9)

for ((sr = 0; sr < ${#sample_rates[@]}; sr++)); do
  for ((l = 0; l < ${#lrs[@]}; l++)); do
    for ((w = 0; w < ${#wds[@]}; w++)); do
      for ((s = 0; s < ${#steps[@]}; s++)); do
        for ((sl = 0; sl < ${#lrs_server[@]}; sl++)); do
          for ((ms = 0; ms < ${#momentums_server[@]}; ms++)); do
            for ((b = 0; b < ${#batch_sizes[@]}; b++)); do
              for k in {1..3}; do
                python federatedscope/main.py --cfg fedhpo/openml/openml_lr.yaml device $cudaid optimizer.lr ${lrs[$l]} optimizer.weight_decay ${wds[$w]} federate.local_update_steps ${steps[$s]} data.type ${dataset}@openml data.batch_size ${batch_sizes[$b]} federate.sample_client_rate ${sample_rates[$sr]} model.out_channels $out_channels federate.share_local_model False federate.online_aggr False fedopt.use True federate.method FedOpt fedopt.lr_server ${lrs_server[$sl]} fedopt.momentum_server ${momentums_server[$ms]} seed $k outdir out_fedopt/lr/${out_dir}_${sample_rates[$sr]} expname lr${lrs[$l]}_wd${wds[$w]}_dropout0_step${steps[$s]}_batch${batch_sizes[$b]}_lrserver${lrs_server[$sl]}_momentumsserver${momentums_server[$ms]}_seed${k} >/dev/null 2>&1
              done
            done
          done
        done
      done
    done
  done
done

echo "HPO ends."
