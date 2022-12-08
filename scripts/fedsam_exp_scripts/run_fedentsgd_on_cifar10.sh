set -e

lda_alpha=$1
cudaid=$2
gamma=$3
eps=$4
lr=$5

echo $lda_alpha
echo $cudaid
echo $gamma
echo $eps
echo $lr

for (( i=0; i<5; i++ ))
do
  python federatedscope/main.py --cfg scripts/fedsam_exp_scripts/fedentsgd_on_cifar10.yaml seed $i device $cudaid data.splitter_args "[{'alpha': ${lda_alpha}}]" trainer.local_entropy.gamma $gamma fedopt.optimizer.lr $gamma trainer.local_entropy.eps $eps train.optimizer.lr $lr expname fedentsgd_${lda_alpha}_${gamma}_${eps}_${lr}_${i}
done
