set -e

cudaid=$1

if [ ! -d "out_dp" ];then
  mkdir out_dp
fi

echo "Starts..."

python flpackage/main.py --cfg flpackage/cv/baseline/fedavg_convnet2_on_femnist.yaml \
device ${cudaid} \
>>out_dp/standard.out \
2>>out_dp/standard.err

echo "Ends."

