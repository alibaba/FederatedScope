set -e

cd ../../..

cudaid=$1
dataset=fs_contest_data

echo "HPO starts..."

models=('gin')
lrs=(0.01 0.1 0.05 0.005 0.5)
local_updates=(1 2 4)
method=FedBN
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
                python federatedscope/main.py --cfg scripts/personalization_exp_scripts/fs_contest/fedbn_gnn_minibatch_on_multi_task.yaml data.batch_size ${bs} device ${cudaid} optimizer.lr ${lrs[$i]} federate.local_update_steps ${local_updates[$j]} model.type ${models[$g]} seed $k outdir ${outdir}/${models[$g]}_${lrs[$i]}_${local_updates[$j]}_bs${bs}_on_${dataset}
            done
        done
    done
done

echo "HPO ends."
