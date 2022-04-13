set -e

cudaid=$1

if [ ! -d "out_fedopt" ];then
  mkdir out_fedopt
fi

echo "Starts..."

python federatedscope/main.py --cfg federatedscope/cv/baseline/fedavg_convnet2_on_femnist.yaml \
device ${cudaid} federate.method fedopt\
>>out_fedopt/fedopt.out \
2>>out_fedopt/fedopt.err

python federatedscope/parse_exp_results.py --input out_fedopt/fedopt.out

echo "Ends."

