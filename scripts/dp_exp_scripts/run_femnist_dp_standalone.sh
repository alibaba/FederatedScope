set -e

cudaid=$1

if [ ! -d "out_nbafl" ];then
  mkdir out_nbafl
fi

echo "NbAFL starts..."

clips=(0.1)
epsilons=(10. 50. 100.)
mus=(0.01)
constants=(1. 2. 3.)

for ((iw=0; iw<${#clips[@]}; iw++ ))
do
    for ((ie=0; ie<${#epsilons[@]}; ie++ ))
    do
        for ((im=0; im<${#mus[@]}; im++ ))
        do
            for ((ic=0; ic<${#constants[@]}; ic++ ))
            do
                python federatedscope/main.py --cfg federatedscope/cv/baseline/fedavg_convnet2_on_femnist.yaml device ${cudaid} nbafl.use True \
                data.root /mnt/gaodawei.gdw/data/ \
                nbafl.mu ${mus[$im]} \
                nbafl.epsilon ${epsilons[$ie]} \
                nbafl.constant ${constants[$ic]} \
                nbafl.w_clip ${clips[$iw]} \
                >>out_nbafl/temp.out \
                2>>out_nbafl/clip_${clips[$iw]}_eps_${epsilons[$ie]}_mu_${mus[$im]}_const_${constants[$ic]}.log
            done
        done
    done
done

for ((iw=0; iw<${#clips[@]}; iw++ ))
do
    for ((ie=0; ie<${#epsilons[@]}; ie++ ))
    do
        for ((im=0; im<${#mus[@]}; im++ ))
        do
            for ((ic=0; ic<${#constants[@]}; ic++ ))
            do
                python federatedscope/../scripts/dp_exp_scripts/parse_nbafl_results.py --input out_nbafl/clip_${clips[$iw]}_eps_${epsilons[$ie]}_mu_${mus[$im]}_const_${constants[$ic]}.log \
                --round 300\
                >>out_nbafl/parse.log
            done
        done
    done
done

echo "Ends."

