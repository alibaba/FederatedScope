set -e

cd ../../../../

cudaid=$1
root=$2
dataset=graph-dt
method=fedavg
outdir=exp_out/${method}
datelog=$(date '+%Y-%m-%d-%H-%M-%S')

if [ ! -d ${outdir} ];then
  mkdir -p ${outdir}
fi

echo "HPO starts..."

log=${outdir}/gin_on_${dataset}_${datelog}.log
for k in {1..3}
do
    python federatedscope/main.py --cfg scripts/B-FHTL_exp_scripts/Graph-DT/hpo/fedavg_gnn_minibatch_on_multi_task.yaml \
    --cfg_client scripts/B-FHTL_exp_scripts/Graph-DT/cfg_per_client.yaml \
    data.root ${root} \
    device ${cudaid} \
    data.type ${dataset} \
    seed $k >>${log} 2>&1
    echo "fedavg k=${k} ends."
done

echo "HPO ends."
