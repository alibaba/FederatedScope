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

log=${outdir}/gin_best-weight_on_${dataset}.log
for k in {1..3}
do
    python federatedscope/main.py --cfg scripts/contest_exp_scripts/fedavg_gnn_minibatch_on_multi_task.yaml \
    --cfg_client scripts/contest_exp_scripts/cfg_per_client_ditto.yaml \
    data.root ${root} \
    device ${cudaid} \
    federate.method Ditto \
    seed $k >>${log} 2>&1
    echo "ditto k=${k} ends."
done

echo "HPO ends."
