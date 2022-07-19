set -e

ENV_NAME = 'org_test'

conda create -n ${ENV_NAME} python=3.9 -y
conda activate ${ENV_NAME}
conda install -y pytorch=1.10.1 torchvision=0.11.2 torchaudio=0.10.1 \
cudatoolkit=11.3 -c pytorch -c conda-forge
python setup.py install