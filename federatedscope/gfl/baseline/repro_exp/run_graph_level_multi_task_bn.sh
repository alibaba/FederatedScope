set -e

cudaid=$1
dname=$2

if [[ $dname = 'mol' ]]; then
    dataset='graph_multi_domain_mol'
elif [[ $dname = 'mix' ]]; then
    dataset='graph_multi_domain_mix'
elif [[ $dname = 'biochem' ]]; then
    dataset='graph_multi_domain_biochem'
elif [[ $dname = 'v1' ]]; then
    dataset='graph_multi_domain_kddcupv1'
else
    dataset='graph_multi_domain_small'
fi

if [ ! -d "out_bn" ];then
  mkdir out_bn
fi

out_channels=0
hidden=64
splitter='ooxx'

echo "HPO starts..."

gnns=('gin')
lrs=(0.01 0.05 0.25)
local_updates=(1 4 16)

for (( g=0; g<${#gnns[@]}; g++ ))
do
    for (( i=0; i<${#lrs[@]}; i++ ))
    do
        for (( j=0; j<${#local_updates[@]}; j++ ))
        do
            for k in {1..5}
            do
                python federatedscope/main.py --cfg federatedscope/gfl/baseline/fedbn_gnn_minibatch_on_multi_task.yaml device ${cudaid} data.type ${dataset} data.splitter ${splitter} optimizer.lr ${lrs[$i]} train.local_update_steps ${local_updates[$j]} model.type ${gnns[$g]} model.out_channels ${out_channels} model.hidden ${hidden} seed $k >>out_bn/${gnns[$g]}_${lrs[$i]}_${local_updates[$j]}_on_${dataset}_${splitter}.log 2>&1
            done
        done
    done
done

for (( g=0; g<${#gnns[@]}; g++ ))
do
    for (( i=0; i<${#lrs[@]}; i++ ))
    do
        for (( j=0; j<${#local_updates[@]}; j++ ))
        do
            python federatedscope/parse_exp_results.py --input out_bn/${gnns[$g]}_${lrs[$i]}_${local_updates[$j]}_on_${dataset}_${splitter}.log
        done
    done
done

echo "HPO ends."
