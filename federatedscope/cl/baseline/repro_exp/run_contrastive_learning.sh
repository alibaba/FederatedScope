set -e

cudaid=$1
method_name=$2
dataset=$3
lda_alpha=$4

cd ../../../..

if [ ! -d "out" ];then
    mkdir out
fi

if [[ $method_name = 'fedgc' ]]; then
    method='fedgc'
    total_round_num='100'
    batch_or_epoch='epoch'
elif [[ $method_name = 'fedsimclr' ]]; then
    method='Fedavg'
    total_round_num='100'
    batch_or_epoch='epoch'
fi

echo "Fed Contrastive Learning starts..."

lrs=(0.003 0.01 0.03)
local_updates=(10)


for (( i=0; i<${#lrs[@]}; i++ ))
do
    for (( j=0; j<${#local_updates[@]}; j++ ))
    do
        for k in {1..2}
        do
            train_yaml=${method_name}_on_${dataset}.yaml
              save_path=${method_name}_on_Cifar4CL_lda${lda_alpha}_lr${lrs[$i]}_lus${local_updates[$j]}_rn${total_round_num}${batch_or_epoch}_seed${k}.ckpt
            python federatedscope/main.py --cfg federatedscope/cl/baseline/${train_yaml} device ${cudaid} federate.save_to ${save_path} federate.total_round_num ${total_round_num} data.splitter_args \[\{\'alpha\'\:${lda_alpha}\}\] train.optimizer.lr ${lrs[$i]} train.local_update_steps ${local_updates[$j]} train.batch_or_epoch ${batch_or_epoch} seed $k>>out/${method_name}_on_Cifar4CL_lda${lda_alpha}_lr${lrs[$i]}_lus${local_updates[$j]}_rn${total_round_num}${batch_or_epoch}.log 2>&1
            linear_prob_yaml=fedcontrastlearning_linearprob_on_cifar10.yaml
             python federatedscope/main.py --cfg federatedscope/cl/baseline/${linear_prob_yaml} device ${cudaid} federate.restore_from ${save_path} data.splitter_args \[\{\'alpha\'\:${lda_alpha}\}\] seed $k>>out/${method_name}_on_Cifar4CL_lda${lda_alpha}_lr${lrs[$i]}_lus${local_updates[$j]}_rn${total_round_num}${batch_or_epoch}.log 2>&1
        done
    done
done


echo "Fed Contrastive Learning ends."
