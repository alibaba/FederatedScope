set -e

cd ../../../../

cudaid=$1
dataset=text-dt
outdir=exp_out/maml
outdir_train=$outdir/train
outdir_ft=$outdir/ft

if [ ! -d ${outdir} ];then
  mkdir -p ${outdir}
fi

if [ ! -d ${outdir_train} ];then
  mkdir -p ${outdir_train}
fi

if [ ! -d ${outdir_ft} ];then
  mkdir -p ${outdir_ft}
fi

echo "HPO starts..."

seeds=(123 1234 12345)

log_train=${outdir_train}/${dataset}.log
log_ft=${outdir_ft}/${dataset}.log
for (( k=0; k<${#seeds[@]}; k++ ))
do
    python federatedscope/main.py \
    --cfg scripts/B-FHTL_exp_scripts/Text-DT/config_maml.yaml \
    --cfg_client scripts/B-FHTL_exp_scripts/Text-DT/config_client_maml.yaml \
    device ${cudaid} \
    outdir $outdir_train/seed_$k \
    federate.save_to $outdir_train/seed_$k/ckpt/global_model.pt \
    data.type ${dataset} \
    seed ${seeds[$k]} >>${log_train} 2>&1

    python federatedscope/main.py \
    --cfg scripts/B-FHTL_exp_scripts/Text-DT/config_ft.yaml \
    --cfg_client scripts/B-FHTL_exp_scripts/Text-DT/config_client_maml_ft.yaml \
    device ${cudaid} \
    outdir $outdir_ft/seed_$k \
    data.type ${dataset} \
    federate.method maml-textdt \
    federate.load_from $outdir_train/seed_$k/ckpt \
    seed ${seeds[$k]} >>${log_ft} 2>&1
done
python federatedscope/parse_exp_results.py --input ${log_ft}

echo "HPO ends."

cd scripts/B-FHTL_exp_scripts/Text-DT/hpo/
