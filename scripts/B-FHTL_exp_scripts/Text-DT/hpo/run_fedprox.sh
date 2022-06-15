set -e

cd ../../../../

cudaid=$1
dataset=text-dt
outdir_raw=exp_out/fedprox

if [ ! -d ${outdir_raw} ];then
  mkdir -p ${outdir_raw}
fi

echo "HPO starts..."

seeds=(123 1234 12345)
mus=(0.001 0.01 0.1)

for (( i=0; i<${#mus[@]}; i++ ))
do
  outdir=$outdir_raw/mu_${mus[$i]}
  if [ ! -d ${outdir} ];then
    mkdir -p ${outdir}
  fi

  log=${outdir}/${dataset}.log
  for (( k=0; k<${#seeds[@]}; k++ ))
  do
      python federatedscope/main.py \
      --cfg scripts/B-FHTL_exp_scripts/Text-DT/config_fedprox.yaml \
      --cfg_client scripts/B-FHTL_exp_scripts/Text-DT/config_client_fedprox.yaml \
      device ${cudaid} \
      outdir $outdir/seed_$k \
      federate.save_to $outdir/seed_$k/ckpt/global_model.pt \
      data.type ${dataset} \
      fedprox.mu ${mus[$i]} \
      seed ${seeds[$k]} >>${log} 2>&1
  done
  python federatedscope/parse_exp_results.py --input ${log}
done

echo "HPO ends."

cd scripts/B-FHTL_exp_scripts/Text-DT/hpo/
