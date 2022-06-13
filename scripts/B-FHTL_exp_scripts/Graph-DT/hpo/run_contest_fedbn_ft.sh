set -e

cd ../../

cudaid=$1
root=./
dataset=fs_contest_data
method=fedbn_ft
outdir=exp_out/${method}

if [ ! -d ${outdir} ];then
  mkdir -p ${outdir}
fi

echo "HPO starts..."

step=1

log=${outdir}/gin_patience-0_lstep-${step}_on_${dataset}.log
for k in {1..3}
do
    python federatedscope/main.py --cfg scripts/B-FHTL_exp_scripts/Graph-DT/fedbn_gnn_minibatch_on_multi_task.yaml \
    --cfg_client scripts/B-FHTL_exp_scripts/Graph-DT/cfg_per_client_bn.yaml \
    data.root ${root} \
    device ${cudaid} \
    data.type ${dataset} \
    trainer.finetune.before_eval True \
    trainer.finetune.steps ${step} \
    seed $k >>${log} 2>&1
    echo "fedbn_ft k=${k} ends."
done

echo "HPO ends."