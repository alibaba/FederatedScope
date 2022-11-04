set -e

cd ..

echo "Run hfl-sgdmf task on movielens1m."

python federatedscope/main.py --cfg federatedscope/mf/baseline/hfl-sgdmf_fedavg_standalone_on_movielens1m.yaml \
          sgdmf.use True \
          sgdmf.epsilon 0.5 \
          sgdmf.delta 0.5 \
          train.optimizer.lr 0.1 \
          train.local_update_steps 20 \
          federate.total_round_num 50 \
          dataloader.batch_size 64
