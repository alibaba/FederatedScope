set -e

# How to use:
#bash scripts/hpo_exp_scripts/uniqueness/fairness/run.sh 8

device=$1


alphas=(0.05 0.5 5.0)
seeds=(12345 12346 12347)
weights=(0.1 1.0 10.0)
trial=0

for (( a=0; a<${#alphas[@]}; a++ ))
do
	alpha=${alphas[$a]}
	for (( s=0; s<${#seeds[@]}; s++ ))
	do
		seed=${seeds[$s]}
		for (( w=0; w<${#weights[@]}; w++ ))
		do
			weight=${weights[$w]}
			folder="${alpha}_${weight}_${seed}_multi_mean_cifar10_avg"
			nohup python federatedscope/hpo.py --cfg scripts/hpo_exp_scripts/uniqueness/fairness/multi_obj.yaml outdir $folder hpo.working_folder $folder device $((trial%${device})) hpo.multi_obj.weight "[${weight}]" seed ${seed} data.splitter_args "[{'alpha': ${alpha}}]" >/dev/null 2>&1 &
			trial=$((trial+1))
			sleep 0.5
		done

		weight=1.0
		folder="${alpha}_${weight}_${seed}_multi_parego_cifar10_avg"
		nohup python federatedscope/hpo.py --cfg scripts/hpo_exp_scripts/uniqueness/fairness/multi_obj.yaml outdir $folder hpo.working_folder $folder device $((trial%${device})) hpo.multi_obj.algo parego hpo.multi_obj.weight "[${weight}]" seed ${seed} data.splitter_args "[{'alpha': ${alpha}}]" >/dev/null 2>&1 &
		trial=$((trial+1))
	done
done