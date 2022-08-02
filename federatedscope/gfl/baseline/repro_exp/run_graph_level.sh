set -e

cudaid=$1
dataset=$2

if [ ! -d "out" ];then
  mkdir out
fi

if [[ $dataset = 'hiv' ]]; then
    out_channels=2
    hidden=64
    splitter='scaffold'
elif [[ $dataset = 'proteins' ]]; then
    out_channels=2
    hidden=64
    splitter='rand_chunk'
elif [[ $dataset = 'imdb-binary' ]]; then
    out_channels=2
    hidden=64
    splitter='graph_type'
else
    out_channels=4
    hidden=1024
fi

echo "HPO starts..."

gnns=('gcn' 'gin' 'gat')
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
                python federatedscope/main.py --cfg federatedscope/gfl/baseline/fedavg_gcn_minibatch_on_hiv.yaml device ${cudaid} data.type ${dataset} data.splitter ${splitter} optimizer.lr ${lrs[$i]} train.local_update_steps ${local_updates[$j]} model.type ${gnns[$g]} model.out_channels ${out_channels} model.hidden ${hidden} seed $k >>out/${gnns[$g]}_${lrs[$i]}_${local_updates[$j]}_on_${dataset}_${splitter}.log 2>&1
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
