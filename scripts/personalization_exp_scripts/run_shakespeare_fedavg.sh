set -e

cd ../..

cudaid=$1
dataset=shakespeare

echo "HPO starts..."

models=('lstm')
lrs=(0.05 0.005 0.5 0.01 0.1)
local_updates=(1 3)
method=FedAvg
bs=64
outdir=exp_out/${method}

for (( g=0; g<${#models[@]}; g++ ))
do
    for (( i=0; i<${#lrs[@]}; i++ ))
    do
        for (( j=0; j<${#local_updates[@]}; j++ ))
        do
            for k in {1..2}
            do
                python flpackage/main.py --cfg flpackage/nlp/baseline/fedavg_lstm_on_shakespeare.yaml federate.method ${method} data.batch_size ${bs} device ${cudaid} optimizer.lr ${lrs[$i]} federate.local_update_steps ${local_updates[$j]} model.type ${models[$g]} seed $k outdir ${outdir}/${models[$g]}_${lrs[$i]}_${local_updates[$j]}_bs${bs}_on_${dataset}
            done
        done
    done
done

echo "HPO ends."
