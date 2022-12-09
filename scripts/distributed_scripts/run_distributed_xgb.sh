set -e

cd ..

echo "Test distributed mode with XGB..."

### server owns global test data
# python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_server.yaml &
### server doesn't own data
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_xgb_server.yaml &
sleep 2

# clients
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_xgb_client_1.yaml &
sleep 2
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_xgb_client_2.yaml &
sleep 2

