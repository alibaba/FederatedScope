set -e

cd ../../

cudaid=$1
root=./
dataset=fs_contest_data
method=maml
outdir=exp_out/${method}
ft_step=10

if [ ! -d ${outdir} ];then
  mkdir -p ${outdir}
fi

echo "HPO starts..."

log=${outdir}/gin_mstep-${ft_step}_change-ilr_on_${dataset}.log
for k in {1..3}
do
    python federatedscope/main.py --cfg scripts/contest_exp_scripts/fedavg_gnn_minibatch_on_multi_task.yaml \
    --cfg_client scripts/contest_exp_scripts/cfg_per_client_maml.yaml \
    data.root ${root} \
    device ${cudaid} \
    data.type ${dataset} \
    trainer.type graphmaml_trainer \
    trainer.finetune.before_eval True \
    trainer.finetune.steps ${ft_step} \
    maml.use True \
    seed $k >>${log} 2>&1
    echo "fedmaml k=${k} ends."
done

echo "HPO ends."