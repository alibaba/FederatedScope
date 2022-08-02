set -e

cudaid=$1
dataset=$2
splitter=$3

if [ ! -d "out" ];then
  mkdir out
fi

if [[ $dataset = 'cora' ]]; then
    out_channels=7
    hidden=64
elif [[ $dataset = 'citeseer' ]]; then
    out_channels=6
    hidden=64
elif [[ $dataset = 'pubmed' ]]; then
    out_channels=5
    hidden=64
else
    out_channels=4
    hidden=1024
fi

echo "HPO starts..."

gnns=('gcn' 'sage' 'gat' 'gpr')
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
                python federatedscope/main.py --cfg federatedscope/gfl/baseline/fedavg_gnn_node_fullbatch_citation.yaml device ${cudaid} data.type ${dataset} data.splitter ${splitter} optimizer.lr ${lrs[$i]} train.local_update_steps ${local_updates[$j]} model.type ${gnns[$g]} model.out_channels ${out_channels} model.hidden ${hidden} seed $k >>out/${gnns[$g]}_${lrs[$i]}_${local_updates[$j]}_on_${dataset}_${splitter}.log 2>&1
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
            python federatedscope/parse_exp_results.py --input out/${gnns[$g]}_${lrs[$i]}_${local_updates[$j]}_on_${dataset}_${splitter}.log
        done
    done
done

echo "HPO ends."
