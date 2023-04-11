#!/system/bin/sh

#================================================================
#   Copyright (C) 2019 Alibaba Ltd. All rights reserved.
#
#   Filename：test_env.sh
#   Author：xxx
#   Date：2019-01-11
#   Description：
#
#================================================================

#pip install
#pip install oss2
#pip install -r requirements.txt

# linter test
#pip install -r requirements/tests.txt
# use internal project for pre-commit due to the network problem
#if [ `git remote -v | grep alibaba  | wc -l` -gt 1 ]; then
#    cp .pre-commit-config.yaml.alibaba  .pre-commit-config.yaml
#fi
#pre-commit run --all-files
#if [ $? -ne 0 ]; then
#    echo "linter test failed, please run 'pre-commit run --all-files' to check"
#    exit -1
#fi

#setup for git-lfs
#if [ ! -e git-lfs/git_lfs.py ]; then
# git submodule init
# git submodule update
#fi

#add ossconfig for git-lfs
#OSS_CONFIG=~/.git_oss_config
#if [ ! -e $OSS_CONFIG ]; then
#    echo "$OSS_CONFIG does not exists"
#    exit
#fi

#download test data
#python git-lfs/git_lfs.py pull
#wget 'http://xxxshare.oss-cn-hangzhou-zmf.aliyuncs.com/gfl%2Ftest_data.tar.gz'
wget 'http://xxxshare.oss-cn-hangzhou-zmf.aliyuncs.com/gfl/test_data.tar.gz'>oss.out 2>oss.err
tar -xzvf test_data.tar.gz>tar.out
mkdir -p test_data/femnist/raw
cp /home/xxx/dev/federatedscope/data/femnist_all_data.zip test_data/femnist/raw
cp -rf /home/xxx/dev/federatedscope/data/MovieLens1M test_data/
#export PYTHONPATH=.
#export TEST_DIR="/tmp/ev_torch_test_${USER}_`date +%s`"

# do not uncomments, casue faild in Online UT, install requirements by yourself on UT machine
# pip install -r requirements.txt
#run test
PYTHONPATH=. python tests/run.py
