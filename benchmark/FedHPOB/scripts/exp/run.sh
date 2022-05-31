nohup bash run_mode.sh cora tabular 0 &
nohup bash run_mode.sh citeseer tabular 0 &
nohup bash run_mode.sh pubmed tabular 0 &

nohup bash run_mode.sh cora raw 0
nohup bash run_mode.sh citeseer raw 1
nohup bash run_mode.sh pubmed raw 2

nohup bash run_mode.sh cora surrogate 0
nohup bash run_mode.sh citeseer surrogate 0
nohup bash run_mode.sh pubmed surrogate 0