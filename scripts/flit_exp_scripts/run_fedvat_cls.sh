set -e

cd ../..

cudaid=$1
dataset=$2
trainer=$3
weightReg=$4
alpha=$5

if [ ! -d "out" ];then
  mkdir out
fi

for k in {1..3}
do
  echo "k=${k}, Trainer=${trainer}, data=${dataset}, weight=${weightReg}, alpha=${alpha} starts..."
  python federatedscope/main.py --cfg federatedscope/gfl/flitplus/fedalgo_cls.yaml device ${cudaid} data.type ${dataset} trainer.type ${trainer} flitplus.weightReg ${weightReg} flitplus.alpha ${alpha} seed ${k} >>out/${trainer}_on_${dataset}_k${k}_weight${weightReg}_alpha${alpha}.log 2>&1
  echo "k=${k}, Trainer=${trainer}, data=${dataset}, weight=${weightReg}, alpha=${alpha} ends."
done
