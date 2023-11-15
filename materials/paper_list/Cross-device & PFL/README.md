## Cross-Device and Personalized Federated Learning
This list is constantly being updated. Feel free to contribute!

### 2023
| Title | Venue | Link | Keywords | Note |
| --- | --- | --- | --- | --- |
| Towards Real-World Cross-Device Federated Learning | KDD | [pdf](https://arxiv.org/abs/2303.13363) | system, cross-Device, heterogeneous device-runtime, FS-Real |
| Personalized Federated Learning with Parameter Propagation | KDD | [pdf](https://dl.acm.org/doi/abs/10.1145/3580305.3599464) | adaptive parameter propagation, selective regularization, FEDORA |
| FedCP: Separating Feature Information for Personalized Federated Learning via Conditional Policy | KDD | [pdf](https://dl.acm.org/doi/abs/10.1145/3580305.3599345) | conditional policy, partial model parameters  |
| Efficient Personalized Federated Learning via Sparse Model-Adaptation | ICML | [pdf](https://arxiv.org/abs/2305.02776) | sparse models, conditional policy, cross-Device, pFedGate  |
| Personalized Federated Learning with Inferred Collaboration Graphs| ICML | [pdf](https://icml.cc/virtual/2023/poster/24966) |  collaboration graph, pFedGraph, poisoning attacks |
| DoCoFL: Downlink Compression for Cross-Device Federated Learning | ICML | [pdf](https://icml.cc/virtual/2023/poster/23954) |  communication compression, cross-device, anchor|
| Personalized Federated Learning with Feature Alignment and Classifier Collaboration | ICLR | [pdf](https://openreview.net/pdf?id=SXZr8aDKia) | Collaboration | feature alignment by regularization, theoretically-guaranteed heads combination
| Test-Time Robust Personalization for Federated Learning | ICLR | [pdf](https://openreview.net/pdf?id=3aBuJEza5sq) | Test-time Robustness |
| A Statistical Framework for Personalized Federated Learning and Estimation: Theory, Algorithms, and Privacy | ICLR | [pdf](https://openreview.net/pdf?id=FUiDMCr_W4o) | Statistical Estimation, Differential Privacy, Empirical/Hierarchical Bayes |
| FiT: Parameter Efficient Few-shot Transfer Learning for Personalized and Federated Image Classification | ICLR | [pdf](https://openreview.net/pdf?id=9aokcgBVIj1) | few-shot learning, transfer learning |
| PerFedMask: Personalized Federated Learning with Optimized Masking Vectors | ICLR | [pdf](https://openreview.net/pdf?id=hxEIgUXLFF) | Masking vectors |
| The Best of Both Worlds: Accurate Global and Personalized Models through Federated Learning with Data-Free Hyper-Knowledge Distillation | ICLR | [pdf](https://openreview.net/pdf?id=29V3AWjVAFi) | Knowledge Distillation, Differential Privacy, | share means of local data representations and soft predictions; no public data

### 2022
| Title | Venue | Link | Keywords | Note |
| --- | --- | --- | --- | --- |
| FedPop: A Bayesian Approach for Personalised Federated Learning | NeurIPS | [pdf](https://arxiv.org/pdf/2206.03611) | Uncertainty quantification; Markov Chain Monte Carlo (MCMC) |
| pFL-Bench: A Comprehensive Benchmark for Personalized Federated Learning | NeurIPS | [pdf](https://arxiv.org/pdf/2206.03655), [code](https://github.com/alibaba/FederatedScope/tree/master/benchmark/pFL-Bench) | Benchmark | 10+ dataset variants, 20+ methods, fruitful metrics and settings.|
| Self-Aware Personalized Federated Learning | NeurIPS | [pdf](https://arxiv.org/pdf/2204.08069.pdf) |Uncertainty quantification; Bayesian hierarchical models|
| Factorized-FL: Personalized Federated Learning with Parameter Factorization & Similarity Matching | NeurIPS | [pdf](https://arxiv.org/pdf/2202.00270) | Kernel Factorization |
| On Sample Optimality in Personalized Collaborative and Federated Learning | NeurIPS | [pdf](https://arxiv.org/pdf/2201.13097) | Information theoretic bounds;  Sample Complexity |
| Personalized Federated Learning towards Communication Efficiency, Robustness and Fairness | NeurIPS | [pdf](https://openreview.net/pdf?id=wFymjzZEEkH) | Infimal convolution; Low-dimensional projection  |
| DisPFL: Towards Communication-Efficient Personalized Federated Learning via Decentralized Sparse Training | ICML | [pdf](https://arxiv.org/pdf/2206.00187.pdf) | Decentralized FL; Sparse Models |
| Personalization Improves Privacy-Accuracy Tradeoffs in Federated Learning | ICML | [pdf](https://arxiv.org/pdf/2202.05318.pdf) | User-level DP; ($\epsilon$, $delta$)-DP  |
| Personalized Federated Learning through Local Memorization | ICML | [pdf](https://arxiv.org/pdf/2111.09360.pdf) | Model interpolation; kNN; |
| Personalized Federated Learning via Variational Bayesian Inference | ICML | [pdf](https://arxiv.org/pdf/2206.07977.pdf) |  Bayesian variational inference; Upper bound |
| Federated Learning with Partial Model Personalization | ICML | [pdf](https://proceedings.mlr.press/v162/pillutla22a/pillutla22a.pdf) | Partial model parameters; Transformer | 
| On Bridging Generic and Personalized Federated Learning for Image Classification | ICLR | [pdf](https://arxiv.org/pdf/2107.00778) |  Partial model parameters; |
| FedBABU: Toward Enhanced Representation for Federated Image Classification | ICLR | [pdf](https://openreview.net/pdf?id=HuaYQfggn5u)  | Partial model parameters; | keep the head (classifer) unchanged during FL training, then conduct fine-tuning before inference
| Towards Personalized Federated Learning | Transactions on Neural Networks and Learning Systems | [pdf](https://arxiv.org/pdf/2103.00710)| Survey |

### 2021
| Title | Venue | Link | Keywords | Note |
| --- | --- | --- | --- | --- |
| Federated muli-task learning under a mixture of distributions  | NeurIPS | [pdf](https://arxiv.org/pdf/2108.10252), [code](https://github.com/omarfoq/FedEM) | Distribution Mixture; Expectation-Maximization; FedEM |
| Parameterized Knowledge Transfer for Personalized Federated Learning  | NeurIPS | [pdf](https://arxiv.org/pdf/2111.02862) | Knowledge Distillation |  transmit only soft-predictions; public dataset required
| Personalized Federated Learning with Gaussian Processes | NeurIPS | [pdf](https://arxiv.org/pdf/2106.15482), [code](https://github.com/IdanAchituve/pFedGP) |  Gaussian process; Generalization bound |
| Ditto: Fair and robust federated learning through personalization  | ICML | [pdf](https://arxiv.org/pdf/2012.04221), [code](https://github.com/litian96/ditto) | Threat model; Fairness; Regularizer |
| Personalized Federated Learning using Hypernetworks  | ICML | [pdf](https://arxiv.org/pdf/2103.04628), [code](https://github.com/AvivSham/pFedHN) | Hypernetwork; Client Embedding |
| Exploiting Shared Representations for Personalized Federated Learning  | ICML | [pdf](https://arxiv.org/pdf/2102.07078.pdf), [code](https://github.com/lgcollins/FedRep) | Partial model parameters | FedRep, shared body (feature extractor), personalized head (classifier)
| Personalized Federated Learning with First Order Model Optimization  | ICLR | [pdf](https://arxiv.org/pdf/2012.08565), [code](https://github.com/NVlabs/FedFomo) | Model mixture |
| FedBN: Federated Learning on Non-IID Features via Local Batch Normalization  | ICLR | [pdf](https://arxiv.org/pdf/2102.07623), [code](https://github.com/med-air/FedBN) | Partial model parameters |


### 2020
| Title | Venue | Link | Keywords | Note |
| --- | --- | --- | --- | --- |
| Personalized federated learning with theoretical guarantees: A model-agnostic meta-learning approach  | NeurIPS | [pdf](https://proceedings.neurips.cc/paper/2020/file/24389bfe4fe2eba8bf9aa9203a44cdad-Paper.pdf) | MAML; Per-FedAvg |
| Personalized federated learning with moreau envelopes  | NeurIPS | [pdf](https://proceedings.neurips.cc/paper/2020/file/f4f1f13c8289ac1b1ee0ff176b56fc60-Paper.pdf), [code](https://github.com/CharlieDinh/pFedMe) | pFedMe; Regularizer; |
| An efficient framework for clustered federated learning  | NeurIPS | [pdf](https://arxiv.org/pdf/2006.04088), [code](https://github.com/jichan3751/ifca) | Iterative clustering; IFCA  |
| Adaptive personalized federated learning  | arXiv | [pdf](https://arxiv.org/pdf/2003.13461), [code](https://github.com/MLOPTPSU/FedTorch) | Model Mixture; APFL  |
| Lower bounds and optimal algorithms for personalized federated learning | NeurIPS | [pdf](https://arxiv.org/pdf/2010.02372)| Communication complexity; |
| Personalized Federated Learning With Differential Privacy  | IEEE Internet of Things Journal | [pdf](https://par.nsf.gov/servlets/purl/10183051)|($\epsilon$, $delta$)-DP  |
| Personalized federated learning for intelligent IoT applications: A cloud-edge based framework | IEEE Open Journal of the Computer Society | [pdf](https://ieeexplore.ieee.org/iel7/8782664/8821528/09090366.pdf)| Device Heterogeneity |
| Survey of Personalization Techniques for Federated Learning  | 2020 Fourth World Conference on Smart Trends in Systems, Security and Sustainability (WorldS4) | [pdf](https://arxiv.org/pdf/2003.08673.pdf)| Survey |  4 pages, 7 types of personalization methods.


### 2019
| Title | Venue | Link | Keywords | Note |
| --- | --- | --- | --- | --- |
| Federated Evaluation of On-device Personalization| arXiv | [pdf](https://arxiv.org/abs/1910.10252) | Personalized Hyper-parameters;  Next-word prediction; RNN| 1. Evaluation scale: tens of millions of users. 2. Evaluation method: testing both global and local models on local test set, calculating and uploading the accuracy delta.