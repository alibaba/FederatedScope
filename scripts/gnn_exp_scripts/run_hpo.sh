set -e

cudaid=$1
dataset=$2

if [ ! -d "hpo_${dataset}" ];then
  mkdir hpo_${dataset}
fi

if [ ! -d "hpo" ];then
  mkdir hpo
fi

rs=(1 2 4 8)
samples=(1 2 4 5)

for (( s=0; s<${#samples[@]}; s++ ))
do
    for (( r=0; r<${#rs[@]}; r++ ))
    do
        for k in {1..5}
        do
            python federatedscope/hpo.py --cfg federatedscope/gfl/baseline/fedavg_gnn_node_fullbatch_citation.yaml federate.sample_client_num ${samples[$s]} device ${cudaid} data.type ${dataset} hpo.r ${rs[$r]} seed $k >>hpo/hpo_on_${dataset}_${rs[$r]}_sample${samples[$s]}.log 2>&1
            rm hpo_${dataset}/*
        done
    done
done