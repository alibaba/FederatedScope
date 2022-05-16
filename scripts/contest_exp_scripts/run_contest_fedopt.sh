set -e

cd ../../..

cudaid=$1
root=$2
dataset=fs_contest_data
method=fedopt
outdir=exp_out/${method}

if [ ! -d ${outdir} ];then
  mkdir -p ${outdir}
fi

echo "HPO starts..."

lrs=(0.01 0.1 0.05 0.005 0.5)
local_updates=(1 2 4)
lrs_s=(0.005 0.01 0.05 0.1 0.5)
mom_s=(0.001 0.01 0.05 0.1 0.5)


for (( i=0; i<${#lrs[@]}; i++ ))
do
    for (( j=0; j<${#local_updates[@]}; j++ ))
    do
        for (( is=0; is<${#lrs_s[@]}; is++ ))
        do
            for (( im=0; im<${#mom_s[@]}; im++ ))
            do
                log=${outdir}/gin_lr-${lrs[$i]}_step-${local_updates[$j]}_slr-${lrs_s[is]}_smom-${mom_s[$im]}_on_${dataset}.log
                for k in {1..3}
                do
                    python federatedscope/main.py --cfg federatedscope/scripts/contest_exp_scripts/fedavg_gnn_minibatch_on_multi_task.yaml \
                    data.root ${root} \
                    device ${cudaid} \
                    data.type ${dataset} \
                    optimizer.lr ${lrs[$i]} \
                    federate.local_update_steps ${local_updates[$j]} \
                    fedopt.use True \
                    fedopt.lr_server ${lrs_s[$is]} \
                    fedopt.momentum_server ${mom_s[$im]} \
                    seed $k >>${log} 2>&1
                done
                python federatedscope/parse_exp_results.py --input ${log}
            done
        done
    done
done

echo "HPO ends."
