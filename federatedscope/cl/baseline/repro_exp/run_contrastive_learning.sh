set -e

cudaid=$1
method_name=$2
dataset=$3

cd ../../../..

if [ ! -d "out" ];then
    mkdir out
fi

if [[ $method_name = 'fedgc' ]]; then
    method='fedgc'
    total_round_num='300'
    batch_or_epoch='batch'
elif [[ $method_name = 'fedsimclr' ]]; then
    method='Fedavg'
    total_round_num='100'
    batch_or_epoch='epoch'
fi

echo "Fed Contrastive Learning starts..."

lrs=(0.01 0.05 0.25)
local_updates=(1 3 5)


for (( i=0; i<${#lrs[@]}; i++ ))
do
    for (( j=0; j<${#local_updates[@]}; j++ ))
    do
        for k in {1..5}
        do
            train_yaml=${method_name}_on_${dataset}.yaml
            save_path=${method_name}_on_Cifar4CL_lda0.5_lr${lrs[$i]}_lus${local_updates[$j]}_rn${total_round_num}${batch_or_epoch}.ckpt
            python federatedscope/main.py --cfg federatedscope/cl/baseline/${train_yaml} device ${cudaid} federate.save_to ${save_path} train.optimizer.lr ${lrs[$i]} train.local_update_steps ${local_updates[$j]} seed $k >>out/${method_name}_on_Cifar4CL_lda0.5_lr${lrs[$i]}_lus${local_updates[$j]}_rn${total_round_num}${batch_or_epoch}.log 2>&1
            linear_prob_yaml=fedcontrastlearning_linearprob_on_cifar10.yaml
             python federatedscope/main.py --cfg federatedscope/cl/baseline/${linear_prob_yaml} device ${cudaid} federate.restore_from ${save_path} >>out/${method_name}_on_Cifar4CL_lda0.5_lr${lrs[$i]}_lus${local_updates[$j]}_rn${total_round_num}${batch_or_epoch}.log 2>&1
        done
    done
done


echo "Fed Contrastive Learning ends."
