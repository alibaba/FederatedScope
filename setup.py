from __future__ import absolute_import, division, print_function

import setuptools

__name__ = 'federatedscope'
__version__ = '0.2.0'
URL = 'https://github.com/rayrayraykk/FederatedScope'

minimal_requires = [
    'numpy==1.21.2', 'scikit-learn==1.0.2', 'scipy==1.7.3', 'pandas==1.4.1',
    'grpcio>=1.45.0', 'grpcio-tools', 'yaml>=5.1', 'fvcore', 'iopath', 'wandb'
    'tensorboard', 'tensorboardX', 'pympler', 'protobuf==3.19.4'
]

full_requires = [
    'pyg==2.0.4', 'rdkit==2021.09.4=py39hccf6a74_0', 'nltk', 'sentencepiece',
    'textgrid', 'typeguard', 'torchtext', 'transformers==4.16.2',
    'tokenizers==0.10.3', 'datasets'
]

test_requires = ['unittest']

dev_requires = test_requires + ['pre-commit']

benchmark_hpo_requires = []

# TODO: add requirement for pfl
benchmark_pfl_requires = []

# TODO: add requirement for htl
benchmark_htl_requires = []

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=__name__,
    version=__version__,
    author="Alibaba Damo Academy",
    author_email="jones.wz@alibaba-inc.com",
    description="Federated learning package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    download_url=f'{URL}/archive/{__version__}.tar.gz',
    keywords=['deep-learning', 'federated-learning', 'benchmark'],
    packages=[
        package for package in setuptools.find_packages()
        if package.startswith(__name__)
    ],
    install_requires=minimal_requires,
    extras_require={
        'full': full_requires,
        'test': test_requires,
        'dev': dev_requires,
        'benchmark_hpo': benchmark_hpo_requires,
        'benchmark_pfl': benchmark_pfl_requires,
        'benchmark_htl': benchmark_htl_requires,
    },
    license="Apache License 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
