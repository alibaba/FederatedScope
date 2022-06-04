set -e

cd ../../

cudaid=$1
root=$2
dataset=fs_contest_data

python federatedscope/main.py --cfg scripts/contest_exp_scripts/fedavg_gnn_minibatch_on_multi_task.yaml \
            --cfg_client scripts/contest_exp_scripts/cfg_per_client.yaml \
            data.root ${root} \
            device ${cudaid} \
            data.type ${dataset} \
            seed 1
