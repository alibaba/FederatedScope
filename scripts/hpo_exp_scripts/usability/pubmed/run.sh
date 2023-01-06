set -e

seed=$1

methods=('rs' 'rs_wrap' 'bo_gp' 'bo_gp_wrap' 'bo_kde' 'bo_kde_wrap' 'bo_rf' 'bo_rf_wrap' 'hb' 'hb_wrap' 'bohb' 'bohb_wrap')


for (( m=0; m<${#methods[@]}; m++ ))
do
	python federatedscope/hpo.py --cfg scripts/hpo_exp_scripts/usability/pubmed/${methods[$m]}.yaml device $((m%4)) seed ${seed}>${methods[$m]}_${seed}.log &
done
