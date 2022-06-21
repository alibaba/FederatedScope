set -e

cd ../..


cudaid=$1
dataset=shakespeare

if [ ! -d "out" ];then
  mkdir out
fi

echo "HPO starts..."

models=('lstm')
lrs=(0.5 0.01 0.1 0.05 0.005)
local_updates=(1 3)
model_num_per_trainer=3
method=FedEM
bs=64
outdir=exp_out/${method}

for k in {1..3}
do
    for (( i=0; i<${#lrs[@]}; i++ ))
    do
        for (( j=0; j<${#local_updates[@]}; j++ ))
        do
            for (( g=0; g<${#models[@]}; g++ ))
            do
                python federatedscope/main.py --cfg federatedscope/nlp/baseline/fedavg_lstm_on_shakespeare.yaml federate.method ${method} data.batch_size ${bs} model.model_num_per_trainer ${model_num_per_trainer} device ${cudaid} train.optimizer.lr ${lrs[$i]} train.local_update_steps ${local_updates[$j]} model.type ${models[$g]} seed $k outdir ${outdir}/${models[$g]}_${lrs[$i]}_${local_updates[$j]}_bs${bs}_on_${dataset}
            done
        done
    done
done

echo "HPO ends."
