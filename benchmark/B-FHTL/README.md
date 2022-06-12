# Benchmark for Federated Hetero-Task Learning

B-FHTL is a benchmark for federated hetero-task learning. 
It contains three federated datasets: Graph-DC, Graph-DT and Text-DT. 
Here we provide the datasets and scripts of the baselines.
More details about the benchmark please refer to [https://arxiv.org/abs/2206.03436](https://arxiv.org/abs/2206.03436)

**Notice**:
Considering FederatedScope is an open-sourced library that updates frequently, to ensure the reproducibility of the experimental results, we create a new branch `feature/B-FHTL`. The users can reproduce the results by running the scripts under the directory `scripts/`, you can also copy the scripts from `benchmark/B-FHTL/${DATA_NAME}`.  

## Dataset
### Graph-DC
The dataset Graph-DC is consisted of 13 graph classification tasks. Each task represents a client in federated learning, and hold different learning goals. You can choose this dataset by setting `data.type` as `fs_contest_data`.

### Graph-DT
The dataset Graph-DT is consisted of 17 graph tasks, including 10 binary classification tasks, 1 multi-class classification task and 5 regression tasks. You can choose this dataset by setting `data.type` as ``.


### Text-DT
The dataset Text-DT is consisted of 3 NLP tasks, including sentiment classification, reading compression and sentence pair similarity prediction. 
You can choose this dataset by setting `data.type` as ``.

## Scripts
We provide some scripts used in the benchmark for users to reproduce the experimental results.
The baselines include 
 - FedAvg
 - FedAvg+FT
 - FedProx
 - FedBn
 - FedBn+FT
 - Ditto
 - FedMaml

for the three datasets in B-FHTL. You can find them under the directory `scripts/${DATA_NAME}/run_${MED_NAME}.sh`, where `DATA_NAME` is the name of the dataset, and `MED_NAME` is chosen from the above baselines.  

To run the script, you should  
- first follow the [README.md](https://github.com/alibaba/FederatedScope/blob/master/README.md) to install the required libraries for FederatedScope,  
- download the datasets and place them in the correct directory (you can specify it by setting `data.root` in the script), 
- run the script by `bash scripts/${DATA_NAME}/run_${MED_NAME}.sh`.