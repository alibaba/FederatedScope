# Intro
We conduct experiments of ~20 pFL method variants on ~10 datasets, to examine their generalization, convergence and efficiency.


We first use wandb sweep to find the best hyper-parameters, then repeat the results three times.
Some running scripts are tracked here.


# Docker
## For remote-debug 
1. enter the container:
`docker run -u root -p 8023:22 --gpus all -it --rm -v "/mnt1:/mnt"  --name your_name-pfl-bench-debug -w /mnt/user_name/FederatedScope alibaba/federatedscope:app-env-torch1.8 /bin/bash
  

2. prepare the ssh and wandb
```bash
apt-get update && apt-get install -y openssh-server
mkdir /var/run/sshd
echo 'root:fsdebug' | chpasswd
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
service ssh start
wandb login --host=http://xx.xx.xx.xx:8080/
```

## For agent docker
0. - sync your codes on the agent machine
    - sync your data on the agent machine, and make sure the you put them in the right path, e.g., `/mnt1/user_name/FederatedScope`
    
   
1. enter the container
`
docker run -u root --gpus all -it --rm -v "/mnt1:/mnt" --name your_name-pfl-bench -w /mnt/user_name/FederatedScope alibaba/federatedscope:app-env-torch1.8 /bin/bash
`
   
2. setup wandb and fs
```bash
wandb login --host=http://xx.xx.xx.xx:8080/
python setup.py install
```
If necessary, patch several packages
`conda install fvcore pympler iopath`


3. run sweep agent, e.g.,
```bash
nohup wandb agent your_name/pFL-bench/xxx &
```
