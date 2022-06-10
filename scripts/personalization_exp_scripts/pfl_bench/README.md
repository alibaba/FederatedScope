# pFL-Bench
The **pFL-Bench** is a comprehensive benchmark for personalized Federated Learning (pFL), which contains more than 10 diverse datasets, 20 competitive pFL baselines, and systematic evaluation with highlighted benefits and potential of pFL. See more details in our [paper](https://arxiv.org/abs/2206.03655).


This repository includes the experimental data, environments, scripts and codes of **pFL-Bench**. We welcome contributions of new pFL methods and datasets to keep pFL-Bench up-to-date and to evolve it! See more details about contribution [here](https://github.com/alibaba/FederatedScope#contributing).


# 1. Data
All the experimental data can be automatically downloaded and processed via *FederatedScope* from the original public data links.

In case the slow or blocked internet connection prevents your downloading, we also provide a public [mirror](https://federatedscope.oss-cn-beijing.aliyuncs.com/pFL-Bench-data.zip) with *aliyun*.
You can download the pre-processed datasets from our public mirror and unzip them in `data` directory under your project root.

If you use other customized data directory, please replace the value of `data.root` in the scripts accordingly.

# 2. Docker Environment
The experiments are conducted on the *federatedscope-torch1.8-application* docker image, you can build it using the [Dockfile](https://github.com/alibaba/FederatedScope/blob/master/enviroment/docker_files/federatedscope-torch1.8-application.Dockerfile). 

We also provide a built docker [image](https://federatedscope.oss-cn-beijing.aliyuncs.com/federatedscope_cuda10_torch18_app.tar), you can download it and creat your image as `docker load < federatedscope_cuda10_torch18_app.tar & docker tag 188b4 alibaba/federatedscope:app-env-torch1.8`

# 3. Run the experiments
We first use wandb sweep to find the best hyper-parameters, then repeat the results three times.
The hyper-parameter search space and hyper-parameter optimization (HPO) scripts for all the methods and all the datasets are in the `scripts/personalization_exp_scripts/pfl_bench` directory and organized by dataset name.

## 3.1 The searched best configs
We put all the config yaml file in the directory `scripts/personalization_exp_scripts/pfl_bench/yaml_best_runs`.
To reproduce the experiments with searched best configurations, you can run the experiment as the following example:
```
python federatedscope/main.py --cfg scripts/personalization_exp_scripts/pfl_bench/yaml_best_runs/FedBN_FEMNIST-s02.yaml
```
Then all the metrics will be tracked in your logfile and send to wandb monitor.

You can customize the yaml file such as your wandb project name, or directly add new config in the command such as 
```
python federatedscope/main.py --cfg scripts/personalization_exp_scripts/pfl_bench/yaml_best_runs/FedBN_FEMNIST-s02.yaml federate.local_update_steps 1
```
More examples for other methods including the combined pFL method variants (e.g., `FedEM-FedBN-FedOPT-FT_cifar10-alpha01.yaml`) are in the directory `scripts/personalization_exp_scripts/pfl_bench/yaml_best_runs`.

## 3.2 Scripts for HPO
We use wandb sweep to find the best hyper-parameters, here are some scripts to do the sweep,

### 3.2.1 For sweep machine
0. login to the wandb host, if you need private hosting, try wandb login [here](https://docs.wandb.ai/guides/self-hosted/local).
1. write your sweep HPO scripts, we provide the full HPO yamls in the `scripts/personalization_exp_scripts/pfl_bench` directory and organized by dataset name. See more details about sweep [here](https://docs.wandb.ai/guides/sweeps).

2. start your sweep by `wandb sweep my_hpo.yaml`, it will print the sweep id such as 
```
wandb: Creating sweep from: sweep_fedAvg_FT.yaml
wandb: Created sweep with ID: mo45xa3d
wandb: View sweep at: http://xx.xx.xxx.xxx:8080/your_sweep_name/pFL-bench/sweeps/mo45xa3d
```


### 3.2.2 For agent machine 
0. - sync your FederatedScope codes to the agent machine
    - sync your data to the agent machine, and make sure you put them in the right path, e.g., `/mnt1/user_name/FederatedScope`
    
   
1. enter the container
`
docker run -u root --gpus all -it --rm -v "/mnt1:/mnt" --name your_name-pfl-bench -w /mnt/user_name/FederatedScope alibaba/federatedscope:app-env-torch1.8 /bin/bash
`
   
2. setup wandb and FederatedScope
```bash
wandb login --host=http://xx.xx.xx.xx:8080/
python setup.py install
```

If necessary, install several missing packages in case of the docker image misses these package
`conda install fvcore pympler iopath`


3. run sweep agent, e.g.,
```bash
nohup wandb agent your_name/pFL-bench/sweep_id &
```

### 3.2.3 For develop/debug machine
For the machine used for remote development and debug
1. enter the container:
```
docker run -u root -p 8023:22 --gpus all -it --rm -v "/mnt1:/mnt"  --name your_name-pfl-bench-debug -w /mnt/user_name/FederatedScope alibaba/federatedscope:app-env-torch1.8 /bin/bash
```

2. prepare the ssh and wandb
```bash
apt-get update && apt-get install -y openssh-server
mkdir /var/run/sshd
echo 'root:fsdebug' | chpasswd
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
service ssh start
wandb login --host=http://xx.xx.xx.xx:8080/
```

3. connect the machine and develop your pFL algorithm
