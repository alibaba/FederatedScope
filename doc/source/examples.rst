Quick Start Examples
====================

We have provided some running examples.
Once FederatedScope has been installed, users are able to directly execute them, e.g.,

train a logistic regression model

.. code-block:: bash
    :linenos:

    python federatedscope/main.py --cfg federatedscope/example_configs/single_process.yaml

    
or train a graph convolutional neural network on our DBLP dataset

.. code-block:: bash
    :linenos:

    python federatedscope/main.py --cfg federatedscope/gfl/baseline/fedavg_on_dblpnew.yaml federate.total_round_num 20
