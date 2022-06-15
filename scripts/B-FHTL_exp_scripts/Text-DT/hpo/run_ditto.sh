set -e

cd ../../../../

cudaid=$1
dataset=text-dt
outdir_raw=exp_out/ditto

if [ ! -d ${outdir_raw} ];then
  mkdir -p ${outdir_raw}
fi

echo "HPO starts..."

seeds=(123 1234 12345)
regular_weights=(0.001 0.01 0.1)

for (( i=0; i<${#regular_weights[@]}; i++ ))
do
  outdir=$outdir_raw/regular_weight_${regular_weights[$i]}
  if [ ! -d ${outdir} ];then
    mkdir -p ${outdir}
  fi

  log=${outdir}/${dataset}.log
  for (( k=0; k<${#seeds[@]}; k++ ))
  do
      python federatedscope/main.py \
      --cfg scripts/B-FHTL_exp_scripts/Text-DT/config_ditto.yaml \
      --cfg_client scripts/B-FHTL_exp_scripts/Text-DT/config_client_ditto.yaml \
      device ${cudaid} \
      outdir $outdir/seed_$k \
      federate.save_to $outdir/seed_$k/ckpt/global_model.pt \
      data.type ${dataset} \
      personalization.regular_weight ${regular_weights[$i]} \
      seed ${seeds[$k]} >>${log} 2>&1
  done
  python federatedscope/parse_exp_results.py --input ${log}
done

echo "HPO ends."

cd scripts/B-FHTL_exp_scripts/Text-DT/hpo/
