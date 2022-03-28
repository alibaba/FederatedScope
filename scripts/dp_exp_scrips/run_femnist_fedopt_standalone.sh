set -e

cudaid=$1

if [ ! -d "out_fedopt" ];then
  mkdir out_fedopt
fi

echo "Starts..."

python flpackage/main.py --cfg flpackage/cv/baseline/fedavg_convnet2_on_femnist.yaml \
device ${cudaid} federate.method fedopt\
>>out_fedopt/fedopt.out \
2>>out_fedopt/fedopt.err

python flpackage/parse_exp_results.py --input out_fedopt/fedopt.out

echo "Ends."

