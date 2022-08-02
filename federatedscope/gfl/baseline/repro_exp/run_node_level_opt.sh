set -e

cudaid=$1
dataset=$2
splitter=$3
gnn=$4
lr=$5
local_update=$6

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

if [[ $gnn = 'gpr' ]]; then
    layer=10
else
    layer=2
fi

echo "HPO starts..."

lr_servers=(0.5 0.1)

for (( s=0; s<${#lr_servers[@]}; s++ ))
do
    for k in {1..5}
    do
        python federatedscope/main.py --cfg federatedscope/gfl/baseline/fedavg_gnn_node_fullbatch_citation.yaml device ${cudaid} data.type ${dataset} data.splitter ${splitter} optimizer.lr ${lr} train.local_update_steps ${local_update} model.type ${gnn} model.out_channels ${out_channels} model.hidden ${hidden} seed $k federate.method FedOpt fedopt.optimizer.lr ${lr_servers[$s]} model.layer ${layer} >>out/${gnn}_${lr}_${local_update}_on_${dataset}_${splitter}_${lr_servers[$s]}_opt.log 2>&1
    done
done

for (( s=0; s<${#lr_servers[@]}; s++ ))
do
    python federatedscope/parse_exp_results.py --input out/${gnn}_${lr}_${local_update}_on_${dataset}_${splitter}_${lr_servers[$s]}_opt.log >>out/final_${gnn}_${lr}_${local_update}_on_${dataset}_${splitter}_opt.out 2>&1
done

echo "HPO ends."
