from __future__ import absolute_import, division, print_function

import setuptools

__name__ = 'federatedscope'
__version__ = '0.1.9'
URL = 'https://github.com/alibaba/FederatedScope'

minimal_requires = [
    'numpy<1.23.0', 'scikit-learn==1.0.2', 'scipy==1.7.3', 'pandas',
    'grpcio>=1.45.0', 'grpcio-tools', 'pyyaml>=5.1', 'fvcore', 'iopath',
    'wandb', 'tensorboard', 'tensorboardX', 'pympler', 'protobuf==3.19.4'
]

test_requires = []

dev_requires = test_requires + ['pre-commit']

benchmark_hpo_requires = [
    'configspace==0.5.0', 'hpbandster==0.7.4', 'smac==1.3.3', 'optuna==2.10.0'
]

# TODO: add requirements for pfl
benchmark_pfl_requires = []

# TODO: add requirements for htl
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
        'test': test_requires,
        'dev': dev_requires,
        'benchmark_hpo': benchmark_hpo_requires,
        'benchmark_pfl': benchmark_pfl_requires,
        'benchmark_htl': benchmark_htl_requires,
    },
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
