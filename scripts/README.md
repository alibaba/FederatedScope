### Attacks in Federated Learning

#### Privacy Attacks
We provide the following four examples to run the membership inference attack, property inference attack, class representative attack and training data/label inference attack, respectively. 

Membership inference attack:

Run the attack in [1]:
```shell script
python federatedscope/main.py --cfg federatedscope/attack/example_attack_config/gradient_ascent_MIA_on_femnist.yaml
```

Property inference attack: Run the BPC [1] attack
```shell script
python federatedscope/main.py --cfg federatedscope/attack/example_attack_config/PIA_toy.yaml
```

Class representative attack: Run DCGAN [2] attack
```shell script
python federatedscope/main.py --cfg federatedscope/attack/example_attack_config/CRA_fedavg_convnet2_on_femnist.yaml
```

Training data/label inference attack: Run the DLG [3] attack 
```shell script
python federatedscope/main.py --cfg federatedscope/attack/example_attack_config/reconstruct_fedavg_opt_on_femnist.yaml
```

[1] Nasr, Milad, R. Shokri and Amir Houmansadr. “Comprehensive Privacy Analysis of Deep Learning: Stand-alone and Federated Learning under Passive and Active White-box Inference Attacks.” ArXiv abs/1812.00910 (2018): n. pag.

[2] Hitaj, Briland, Giuseppe Ateniese, and Fernando Perez-Cruz. "Deep models under the GAN: information leakage from collaborative deep learning." Proceedings of the 2017 ACM SIGSAC conference on computer and communications security. 2017

[3] Zhu, Ligeng, Zhijian Liu, and Song Han. "Deep leakage from gradients." Advances in Neural Information Processing Systems 32 (2019).




