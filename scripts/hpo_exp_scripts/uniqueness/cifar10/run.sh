set -e

# How to use:
#bash scripts/hpo_exp_scripts/uniqueness/cifar10/run.sh 0.05 12345
#bash scripts/hpo_exp_scripts/uniqueness/cifar10/run.sh 0.05 12346
#bash scripts/hpo_exp_scripts/uniqueness/cifar10/run.sh 0.05 12347
#
#bash scripts/hpo_exp_scripts/uniqueness/cifar10/run.sh 0.5 12345
#bash scripts/hpo_exp_scripts/uniqueness/cifar10/run.sh 0.5 12346
#bash scripts/hpo_exp_scripts/uniqueness/cifar10/run.sh 0.5 12347
#
#bash scripts/hpo_exp_scripts/uniqueness/cifar10/run.sh 5.0 12345
#bash scripts/hpo_exp_scripts/uniqueness/cifar10/run.sh 5.0 12346
#bash scripts/hpo_exp_scripts/uniqueness/cifar10/run.sh 5.0 12347

alpha=$1
seed=$2

methods=('rs' 'rs_wrap' 'bo_gp' 'bo_gp_wrap' 'bo_kde' 'bo_kde_wrap' 'bo_rf' 'bo_rf_wrap' 'hb' 'hb_wrap' 'bohb' 'bohb_wrap')


for (( m=0; m<${#methods[@]}; m++ ))
do
	nohup python federatedscope/hpo.py --cfg scripts/hpo_exp_scripts/uniqueness/cifar10/${methods[$m]}.yaml device $((m%4)) seed ${seed} data.splitter_args "[{'alpha': ${alpha}}]" >/dev/null 2>&1 &
done
