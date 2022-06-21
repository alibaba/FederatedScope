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

if [[ $dataset = 'fb15k-237' ]]; then
    out_channels=237
    hidden=64
elif [[ $dataset = 'wn18' ]]; then
    out_channels=18
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

mu=(0.1 1.0 5.0)

for (( s=0; s<${#mu[@]}; s++ ))
do
    for k in {1..5}
    do
        python federatedscope/main.py --cfg federatedscope/gfl/baseline/fedavg_gcn_fullbatch_on_kg.yaml device ${cudaid} data.type ${dataset} data.splitter ${splitter} optimizer.lr ${lr} train.local_update_steps ${local_update} model.type ${gnn} model.out_channels ${out_channels} model.hidden ${hidden} seed $k fedprox.use True fedprox.mu ${mu[$s]} model.layer ${layer} >>out/${gnn}_${lr}_${local_update}_on_${dataset}_${splitter}_${mu[$s]}_prox.log 2>&1
    done
done

for (( s=0; s<${#mu[@]}; s++ ))
do
    python federatedscope/parse_exp_results.py --input out/${gnn}_${lr}_${local_update}_on_${dataset}_${splitter}_${mu[$s]}_prox.log >>out/final_${gnn}_${lr}_${local_update}_on_${dataset}_${splitter}_prox.out 2>&1
done

echo "HPO ends."
