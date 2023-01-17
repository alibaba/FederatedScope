set -e

#bash scripts/hpo_exp_scripts/robustness/attack/run.sh "[1,2,3,4]" 1.0 12345 8
#bash scripts/hpo_exp_scripts/robustness/attack/run.sh "[1,2,3,4]" 1.0 12346 8
#bash scripts/hpo_exp_scripts/robustness/attack/run.sh "[1,2,3,4]" 1.0 12347 8
#
#bash scripts/hpo_exp_scripts/robustness/attack/run.sh "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]" 1.0 12345 8
#bash scripts/hpo_exp_scripts/robustness/attack/run.sh "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]" 1.0 12346 8
#bash scripts/hpo_exp_scripts/robustness/attack/run.sh "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]" 1.0 12347 8
#
#bash scripts/hpo_exp_scripts/robustness/attack/run.sh "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64]" 1.0 12345 8
#bash scripts/hpo_exp_scripts/robustness/attack/run.sh "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64]" 1.0 12346 8
#bash scripts/hpo_exp_scripts/robustness/attack/run.sh "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64]" 1.0 12347 8

attacker=$1
sigma=$2
seed=$3
device=$4

methods=('rs_wrap' 'bo_gp_wrap' 'bo_kde_wrap' 'bo_rf_wrap' 'hb_wrap' 'bohb_wrap')


for (( m=0; m<${#methods[@]}; m++ ))
do
	nohup python federatedscope/hpo.py --cfg scripts/hpo_exp_scripts/robustness/attack/${methods[$m]}.yaml device $((m%${device})) seed ${seed} hpo.fedex.attack.id ${attacker} hpo.fedex.attack.sigma ${sigma} >/dev/null 2>&1 &
done
