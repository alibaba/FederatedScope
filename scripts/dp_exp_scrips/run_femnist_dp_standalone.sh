set -e

cudaid=$1

if [ ! -d "out_dp" ];then
  mkdir out_dp
fi

echo "DP starts..."

clips=(0.1)
epsilons=(100.)
mus=(0.1)
constants=(2. 3. 4.)

for ((iw=0; iw<${#clips[@]}; iw++ ))
do
    for ((ie=0; ie<${#epsilons[@]}; ie++ ))
    do
        for ((im=0; im<${#mus[@]}; im++ ))
        do
            for ((ic=0; ic<${#constants[@]}; ic++ ))
            do
                python federatedscope/main.py --cfg federatedscope/cv/baseline/fedavg_convnet2_on_femnist.yaml device ${cudaid} nbafl.use True \
                nbafl.mu ${mus[$im]} \
                nbafl.epsilon ${epsilons[$ie]} \
                nbafl.constant ${constants[$ic]} \
                nbafl.w_clip ${clips[$iw]} \
                >>out_dp/clip_${clips[$iw]}_eps_${epsilons[$ie]}_mu_${mus[$im]}_const_${constants[$ic]}.out \
                2>>out_dp/clip_${clips[$iw]}_eps_${epsilons[$ie]}_mu_${mus[$im]}_const_${constants[$ic]}.err
            done
        done
    done
done

echo "Ends."

