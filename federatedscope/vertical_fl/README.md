### Vertical Federated Learning

We provide an example for vertical federated learning, you run run with:
```bash
python3 ../main.py --cfg vertical_fl.yaml
```

You can specify customized configurations in `vertical_fl.yaml`, such as `data.type` and `federate.total_round_num`. 
More details of the provided example can be found in [Tutorial](https://federatedscope.io/docs/cross-silo/).

Note that FederatedScope only provide an `abstract_paillier`, user can refer to [pyphe](https://github.com/data61/python-paillier/blob/master/phe/paillier.py) for the detail implementation, or adopt other homomorphic encryption algorithms.

More support for vertical federated learning is coming soon! We really appreciate any contributions to FederatedScope !