# The federatedscope image includes all runtime stuffs of federatedscope,
# with customized miniconda and required packages installed.

# based on the nvidia-docker
# NOTE: please pre-install the NVIDIA drivers and `nvidia-docker2` in the host machine,
# see details in https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
ARG ROOT_CONTAINER=nvidia/cuda:11.7.0-runtime-ubuntu20.04

FROM $ROOT_CONTAINER

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# shanghai zoneinfo
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install basic tools
RUN apt-get -y update \
    && apt-get -y install curl git gcc g++ make openssl libssl-dev libbz2-dev libreadline-dev libsqlite3-dev python-dev libmysqlclient-dev

# install miniconda,  in batch (silent) mode, does not edit PATH or .bashrc or .bash_profile
RUN apt-get update -y \
    && apt-get install -y wget
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh \
    && bash Miniconda3-py39_23.1.0-1-Linux-x86_64.sh -b \
    && rm Miniconda3-py39_23.1.0-1-Linux-x86_64.sh

ENV PATH=/root/miniconda3/bin:${PATH}
RUN source activate

RUN conda update -y conda \
    && conda config --add channels conda-forge

# Install torch
RUN conda install -y pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia \
		&& conda clean -a -y

# Install FS-LLM
RUN cd /root \
		&& git clone -b llm https://github.com/alibaba/FederatedScope.git \
		&& cd /root/FederatedScope \
		&& pip install -e .[llm] \
		&& pip cache purge

# Prepare datas
RUN mkdir /root/FederatedScope/data \
		&& cd /root/FederatedScope/data \
		&& wget https://raw.githubusercontent.com/databrickslabs/dolly/d000e3030970379aabbf6d291f50ffdd3b715b64/data/databricks-dolly-15k.jsonl \
		&& wget https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl -O gsm8k_train.jsonl \
		&& wget https://raw.githubusercontent.com/openai/grade-school-math/2909d34ef28520753df82a2234c357259d254aa8/grade_school_math/data/test.jsonl -O gsm8k_test.jsonl \
		&& wget https://raw.githubusercontent.com/sahil280114/codealpaca/d269da106a579a623a654529b3cb91b5dfa9c72f/data/rosetta_alpaca.json

# Prepare Evaluation
RUN cd /root/FederatedScope \
		&& git clone https://github.com/openai/human-eval \
		&& pip install -e human-eval \
		&& pip cache purge