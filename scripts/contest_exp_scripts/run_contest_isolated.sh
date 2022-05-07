set -e

cudaid=$1
exp=$2

dataset=fs_contest_data

if [ ! -d "out_${exp}" ];then
  mkdir out_${exp}
fi

out_channels=0
hidden=64
splitter='ooxx'

echo "HPO starts..."

gnns=('gin')
lrs=(0.01 0.1 0.05 0.005 0.5)
local_updates=(1 2 4)
patiences=(1 5 10)

for (( g=0; g<${#gnns[@]}; g++ ))
do
    for (( i=0; i<${#lrs[@]}; i++ ))
    do
        for (( j=0; j<${#local_updates[@]}; j++ ))
        do
            for (( p=0; p<${#patiences[@]}; p++ ))
            do
                for k in {1..5}
                do
                    python federatedscope/main.py --cfg federatedscope/gfl/baseline/fedavg_gnn_minibatch_on_multi_task.yaml\
                     data.root data/ \
                     device ${cudaid} \
                     data.type ${dataset} \
                     data.splitter ${splitter} \
                     optimizer.lr ${lrs[$i]} \
                     federate.local_update_steps ${local_updates[$j]} \
                     model.type ${gnns[$g]} \
                     model.out_channels ${out_channels} \
                     model.hidden ${hidden} \
                     federate.method local \
                     early_stop.patience ${patiences[$p]} \
                     seed $k >>out_${exp}/${gnns[$g]}_${lrs[$i]}_${local_updates[$j]}_${patiences[$p]}_on_${dataset}_${splitter}.log 2>&1
                done
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
            for (( p=0; p<${#patiences[@]}; p++ ))
            do
                python federatedscope/parse_exp_results.py --input out_${exp}/${gnns[$g]}_${lrs[$i]}_${local_updates[$j]}_${patiences[$p]}_on_${dataset}_${splitter}.log
            done
        done
    done
done

echo "HPO ends."
