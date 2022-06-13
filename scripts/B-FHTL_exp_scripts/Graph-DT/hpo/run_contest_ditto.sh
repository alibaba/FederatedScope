set -e

cd ../../

cudaid=$1
root=./
dataset=fs_contest_data
method=ditto
outdir=exp_out/${method}

if [ ! -d ${outdir} ];then
  mkdir ${outdir}
fi

echo "HPO starts..."

personalization_regular_weight=0.01

log=${outdir}/gin_weight-${personalization_regular_weight}_on_${dataset}.log
for k in {1..3}
do
    python federatedscope/main.py --cfg scripts/B-FHTL_exp_scripts/Graph-DT/fedavg_gnn_minibatch_on_multi_task.yaml \
    --cfg_client scripts/B-FHTL_exp_scripts/Graph-DT/cfg_per_client_ditto.yaml \
    data.root ${root} \
    device ${cudaid} \
    federate.method Ditto \
    personalization.regular_weight ${personalization_regular_weight} \
    seed $k >>${log} 2>&1
    echo "ditto k=${k} ends."
done

echo "HPO ends."
