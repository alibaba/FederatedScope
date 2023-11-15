set -e

lda_alpha=$1
cudaid=$2
gamma=$3
lr=$4
eps=$5
alpha=$6
annealing=$7


echo $lda_alpha
echo $cudaid
echo $gamma
echo $lr
echo $eps
echo $alpha
echo $annealing

for (( i=0; i<5; i++ ))
do
  CUDA_VISIBLE_DEVICES="${cudaid}" python federatedscope/main.py --cfg scripts/wide_valley_exp_scripts/fedentsgd_on_cifar10.yaml seed $i data.splitter_args "[{'alpha': ${lda_alpha}}]" trainer.local_entropy.gamma $gamma fedopt.optimizer.lr 1.0 fedopt.annealing $annealing trainer.local_entropy.eps $eps trainer.local_entropy.alpha $alpha train.optimizer.lr $lr expname fedentsgd_${lda_alpha}_${gamma}_${eps}_${annealing}_${i}
done
