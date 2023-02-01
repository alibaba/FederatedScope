set -e

cd ..

echo "Test distributed mode with LR..."

### server owns global test data
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_server.yaml &
### server doesn't own data
# python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_server_no_data.yaml &
sleep 2

# clients
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_client_1.yaml &
sleep 2
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_client_2.yaml &
sleep 2
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_client_3.yaml &

