set -e

NODE=$1

echo "Test distributed mode with LR..."

echo "Data generation"
python scripts/gen_data.py

if [ $NODE -eq 0 ]
then
  python federatedscope/main.py --cfg federatedscope/example_configs/distributed_femnist_server.yaml distribute.server_host 172.17.138.149 &
  sleep 2
  python federatedscope/main.py --cfg federatedscope/example_configs/distributed_femnist_client_1.yaml distribute.server_host 172.17.138.149 distribute.client_host 172.17.138.149 distribute.client_port 50052 &
else
  python federatedscope/main.py --cfg federatedscope/example_configs/distributed_femnist_client2.yaml distribute.server_host 172.17.138.149 distribute.server_port 50051 distribute.client_host 172.17.138.148 distribute.client_port 50053 &
  sleep 2
  python federatedscope/main.py --cfg federatedscope/example_configs/distributed_femnist_client2.yaml distribute.server_host 172.17.138.149 distribute.server_port 50051 distribute.client_host 172.17.138.148 distribute.client_port 50054 &
fi
