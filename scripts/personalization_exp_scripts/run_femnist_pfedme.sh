set -e

cd ../..

cudaid=$1
dataset=femnist

echo "HPO starts..."

models=('convnet2')
lrs=(0.05 0.005 0.5 0.01 0.1)
local_updates=(1 3)
personalization_lr=-1.0
personalization_beta=1.0
personalization_K=3
method=pFedMe
bs=64
outdir=exp_out/${method}

for (( g=0; g<${#models[@]}; g++ ))
do
    for (( i=0; i<${#lrs[@]}; i++ ))
    do
        for (( j=0; j<${#local_updates[@]}; j++ ))
        do
            for k in {1..3}
            do
                python federatedscope/main.py --cfg federatedscope/cv/baseline/fedavg_convnet2_on_femnist.yaml federate.method ${method} dataloader.batch_size ${bs} personalization.K ${personalization_K} personalization.lr ${personalization_lr} personalization.regular_weight ${personalization_beta} device ${cudaid} optimizer.lr ${lrs[$i]} train.local_update_steps ${local_updates[$j]} model.type ${models[$g]} seed $k outdir ${outdir}/${models[$g]}_${lrs[$i]}_${local_updates[$j]}_bs${bs}_on_${dataset}
            done
        done
    done
done

echo "HPO ends."
