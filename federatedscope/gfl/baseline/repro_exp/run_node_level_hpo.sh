set -e

cudaid=$1
dataset=$2
gnn='gcn'

if [ ! -d "out" ];then
  mkdir out
fi

if [[ $dataset = 'cora' ]]; then
    out_channels=7
    hidden=64
    
    num=16
    arry1=(0.5 0.0 0.25 16)
    arry2=(0.5 0.0 0.01 16)
    arry3=(0.0 0.0 0.01 1)
    arry4=(0.0 0.0005 0.01 1)
    arry5=(0.5 0.0 0.01 4)
    arry6=(0.5 0.0 0.01 1)
    arry7=(0.0 0.0 0.25 16)
    arry8=(0.0 0.0005 0.25 1)
    arry9=(0.5 0.0 0.05 1)
    arry10=(0.5 0.0 0.25 4)
    arry11=(0.0 0.0005 0.25 4)
    arry12=(0.0 0.0005 0.01 4)
    arry13=(0.5 0.0 0.25 1)
    arry14=(0.0 0.0005 0.25 16)
    arry15=(0.5 0.0 0.05 4)
    arry16=(0.0 0.0005 0.01 16)
    
elif [[ $dataset = 'citeseer' ]]; then
    out_channels=6
    hidden=64
    
    num=20
    arry1=(0.5 0.0 0.01 4)
    arry2=(0.0 0.0005 0.01 1)
    arry3=(0.5 0.0 0.05 4)
    arry4=(0.5 0.0 0.25 1)
    arry5=(0.0 0.0 0.05 16)
    arry6=(0.0 0.0005 0.01 4)
    arry7=(0.0 0.0005 0.05 1)
    arry8=(0.0 0.0005 0.25 4)
    arry9=(0.0 0.0005 0.05 16)
    arry10=(0.0 0.0005 0.25 1)
    arry11=(0.0 0.0 0.25 4)
    arry12=(0.0 0.0 0.25 16)
    arry13=(0.5 0.0 0.05 16)
    arry14=(0.5 0.0 0.01 16)
    arry15=(0.0 0.0 0.01 1)
    arry16=(0.5 0.0 0.01 1)
    arry17=(0.0 0.0005 0.05 4)
    arry18=(0.0 0.0 0.25 1)
    arry19=(0.0 0.0005 0.01 16)
    arry20=(0.0 0.0 0.05 4)
    
elif [[ $dataset = 'pubmed' ]]; then
    out_channels=5
    hidden=64
    
    num=15
    arry1=(0.5 0.0 0.05 1)
    arry2=(0.5 0.0 0.01 16)
    arry3=(0.0 0.0005 0.25 16)
    arry4=(0.0 0.0005 0.01 4)
    arry5=(0.5 0.0 0.25 4)
    arry6=(0.5 0.0 0.25 16)
    arry7=(0.0 0.0 0.25 4)
    arry8=(0.0 0.0 0.01 1)
    arry9=(0.0 0.0 0.05 1)
    arry10=(0.0 0.0005 0.05 1)
    arry11=(0.5 0.0 0.01 4)
    arry12=(0.0 0.0 0.01 4)
    arry13=(0.5 0.0 0.25 1)
    arry14=(0.0 0.0005 0.01 1)
    arry15=(0.5 0.0 0.01 1)
 
else
    out_channels=4
    hidden=1024
fi

echo "HPO starts..."


for (( i=1; i<num+1; i++ ))
do
    eval dropout=\${arry${i}[0]}
    eval wd=\${arry${i}[1]}
    eval lr=\${arry${i}[2]}
    eval local_update=\${arry${i}[3]}
    for k in {1..5}
    do
        python federatedscope/main.py --cfg federatedscope/gfl/baseline/fedavg_gnn_node_fullbatch_citation.yaml device ${cudaid} data.type ${dataset} model.dropout ${dropout} optimizer.weight_decay ${wd} optimizer.lr ${lr} train.local_update_steps ${local_update} model.type ${gnn} model.out_channels ${out_channels} model.hidden ${hidden} seed $k >>out/${gnn}_${dropout}_${wd}_${lr}_${local_update}_on_${dataset}.log 2>&1
    done
done

for (( i=1; i<num+1; i++ ))
do
    eval dropout=\${arry${i}[0]}
    eval wd=\${arry${i}[1]}
    eval lr=\${arry${i}[2]}
    eval local_update=\${arry${i}[3]}
    python federatedscope/parse_exp_results.py --input out/${gnn}_${dropout}_${wd}_${lr}_${local_update}_on_${dataset}.log
done

echo "HPO ends."
