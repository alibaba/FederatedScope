set -e

cd ../../

cudaid=$1
root=$2
dataset=fs_contest_data
method=fedbn
outdir=exp_out/${method}

if [ ! -d ${outdir} ];then
  mkdir -p ${outdir}
fi

echo "HPO starts..."

log=${outdir}/gin_on_${dataset}.log
for k in {1..3}
do
    python federatedscope/main.py --cfg scripts/B-FHTL_exp_scripts/Graph-DT/fedbn_gnn_minibatch_on_multi_task.yaml \
    --cfg_client scripts/B-FHTL_exp_scripts/Graph-DT/cfg_per_client_bn.yaml \
    data.root ${root} \
    device ${cudaid} \
    data.type ${dataset} \
    seed $k >>${log} 2>&1
    echo "fedbn k=${k} ends."
done

echo "HPO ends."
