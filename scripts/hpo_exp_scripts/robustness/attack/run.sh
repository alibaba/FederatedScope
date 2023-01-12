set -e

attacker=$1
sigma=$2
seed=$3

methods=('rs_wrap' 'bo_gp_wrap' 'bo_kde_wrap' 'bo_rf_wrap' 'hb_wrap' 'bohb_wrap')


for (( m=0; m<${#methods[@]}; m++ ))
do
	nohup python federatedscope/hpo.py --cfg scripts/hpo_exp_scripts/robustness/attack/${methods[$m]}.yaml device $((m%4)) seed ${seed} hpo.fedex.attack.id ${attacker} hpo.fedex.attack.sigma ${sigma} >/dev/null 2>&1 &
done
