# The federatedscope image includes all runtime stuffs of federatedscope,
# with customized miniconda and required packages installed.

# based on the nvidia-docker
# NOTE: please pre-install the NVIDIA drivers and `nvidia-docker2` in the host machine,
# see details in https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
ARG HELM_SOURCE=https://github.com/DavdGao/helm.git
ARG HELM_BRANCH=main
ARG FS_SOURCE=https://github.com/alibaba/FederatedScope.git
ARG FS_BRANCH=dev/llm
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
RUN conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia \
    && conda clean -a -y

# Install helm
RUN pip install -e git+${HELM_SOURCE}@${HELM_BRANCH}#egg=crfm-helm

# Install fs
RUN pip install -e git+${FS_SOURCE}@${FS_BRANCH}#egg=federatedscope