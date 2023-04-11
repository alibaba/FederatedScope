from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="federatedscope",
    version="0.1.0",
    author="Alibaba Damo Academy",
    author_email="",
    description="Federated learning package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=[
        package for package in setuptools.find_packages()
        if package.startswith('federatedscope')
    ],
    install_requires=[
        'torch', 'networkx', 'numpy', 'grpcio>=1.45.0', 'grpcio-tools'
    ],
    setup_requires=[],
    extras_require={'yaml': ['yaml>=5.1']},
    license="Apache License 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    test_suite='nose.collector',
    test_require=['nose'],
    python_requires='>=3.9',
)
