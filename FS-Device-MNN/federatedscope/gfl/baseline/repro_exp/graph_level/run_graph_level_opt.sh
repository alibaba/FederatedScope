set -e

cudaid=$1
dataset=$2
gnn=$3
lr=$4
local_update=$5

cd ../../../../..

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
    splitter='lda'
else
    out_channels=4
    hidden=1024
fi


echo "HPO starts..."

lr_servers=(0.5 0.1)

for (( s=0; s<${#lr_servers[@]}; s++ ))
do
    for k in {1..5}
    do
        python federatedscope/main.py --cfg federatedscope/gfl/baseline/fedavg_gcn_minibatch_on_hiv.yaml device ${cudaid} data.type ${dataset} data.splitter ${splitter} train.optimizer.lr ${lr} train.local_update_steps ${local_update} model.type ${gnn} model.out_channels ${out_channels} model.hidden ${hidden} seed $k fedopt.use True fedopt.optimizer.lr ${lr_servers[$s]} >>out/${gnn}_${lr}_${local_update}_on_${dataset}_${splitter}_${lr_servers[$s]}_opt.log 2>&1
    done
done

echo "HPO ends."
