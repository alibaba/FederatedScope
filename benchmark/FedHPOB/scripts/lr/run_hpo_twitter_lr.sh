set -e

cudaid=$1
dataset=$2

cd ../..

out_dir=out_${dataset} 

if [ ! -d $out_dir ];then
  mkdir $out_dir
fi

echo "HPO starts..."

sample_rates=(0.01)
lrs=(0.00001 0.0001 0.001 0.01 0.1 1.0)
wds=(0.0 0.001 0.01 0.1)
steps=(1 2 3 4)
batch_sizes=(64)

for (( sr=0; sr<${#sample_rates[@]}; sr++ ))
do
    for (( l=0; l<${#lrs[@]}; l++ ))
    do
        for (( w=0; w<${#wds[@]}; w++ ))
        do
            for (( s=0; s<${#steps[@]}; s++ ))
            do
                for (( b=0; b<${#batch_sizes[@]}; b++ ))
                do
                    for k in {1..3}
                    do
                        python federatedscope/main.py --cfg fedhpo/openml/openml_lr.yaml device $cudaid optimizer.lr ${lrs[$l]} optimizer.weight_decay ${wds[$w]} federate.local_update_steps ${steps[$s]} data.type ${dataset}@openml data.batch_size ${batch_sizes[$b]} federate.sample_client_rate ${sample_rates[$sr]} model.out_channels $out_channels seed $k outdir lr/${out_dir}_${sample_rates[$sr]} expname lr${lrs[$l]}_wd${wds[$w]}_dropout0_step${steps[$s]}_batch${batch_sizes[$b]}_seed${k} >/dev/null 2>&1
                    done
                done
            done
        done
    done
done

echo "HPO ends."
