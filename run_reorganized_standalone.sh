set -e

cudaid=$1

if [ ! -d "out_reorganize" ];then
  mkdir out_reorganize
fi

echo "Starts..."

lr=0.01

python federatedscope/main.py --cfg federatedscope/example_configs/single_process.yaml device ${cudaid} data.type toy data.splitter ooxx \
    optimizer.lr ${lr} model.type lr federate.mode standalone trainer.type general federate.total_round_num 50 \
    >>out_reorganize/lr.out \
     2>>out_reorganize/lr.err

echo "Ends."
