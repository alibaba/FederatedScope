set -e

cd ..

echo "Test distributed mode with XGB..."

### server
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_xgb_server.yaml &
sleep 2

# clients
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_xgb_client_1.yaml &
sleep 2
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_xgb_client_2.yaml &
sleep 2

