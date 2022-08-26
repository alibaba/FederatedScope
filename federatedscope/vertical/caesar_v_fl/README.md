### Caesar Vertical Federated Learning

We provide an example for seCure lArge-scalE SpArse logistic Regression (caesar) vertical federated learning, you can run with:
```bash
python3 ../../main.py --cfg caesar_v_fl.yaml
```

Implementation of caesar vertical FL refer to `When Homomorphic Encryption
    Marries Secret Sharing: Secure Large-Scale Sparse Logistic Regression and
    Applications in Risk Control` [Chen, et al., 2021]
    (https://arxiv.org/abs/2008.08753)

You can specify customized configurations in `caesar_v_fl.yaml`, such as `data.type` and `federate.total_round_num`. 


Note that FederatedScope only provide an `abstract_paillier`, user can refer to [pyphe](https://github.com/data61/python-paillier/blob/master/phe/paillier.py) for the detail implementation, or adopt other homomorphic encryption algorithms.

More support for vertical federated learning is coming soon! We really appreciate any contributions to FederatedScope !