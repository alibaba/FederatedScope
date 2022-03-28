set -e

echo "Test distributed mode with LR..."

echo "Data generation"
python scripts/gen_data.py

python flpackage/main.py --cfg flpackage/example_configs/distributed_server.yaml &
sleep 2
python flpackage/main.py --cfg flpackage/example_configs/distributed_client_1.yaml &
sleep 2
python flpackage/main.py --cfg flpackage/example_configs/distributed_client_2.yaml &

