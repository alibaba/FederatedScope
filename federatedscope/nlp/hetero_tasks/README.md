## Collaborating Heterogeneous Natural Language Processing Tasks via Federated Learning
The implementation of *Collaborating Heterogeneous Natural Language Processing Tasks via Federated Learning*.

In this study, we further broaden the application scope of FL in NLP by proposing an *Assign-Then-Contrast* (denoted as **ATC**) framework, which enables clients with heterogeneous NLP tasks to construct an FL course and learn useful knowledge from each other.

## Installation
First of all, you need to install FederatedScope, please refer to [installation](https://github.com/alibaba/FederatedScope#step-1-installation).

Besides, we need some additional requirements for NLP tasks, including:
* NLTK
* Transformers
* ROUGE
* METEOR

```bash
# Install NLTK and Transformers
pip install nltk
pip install transformers==4.21.0

# Install ROUGE
git clone https://github.com/bheinzerling/pyrouge
cd pyrouge
pip install -e .
git clone https://github.com/andersjo/pyrouge.git rouge
pyrouge_set_rouge_path $(realpath rouge/tools/ROUGE-1.5.5/)
sudo apt-get install libxml-parser-perl
cd rouge/tools/ROUGE-1.5.5/data
rm WordNet-2.0.exc.db
./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db
python -m pyrouge.test

# Download METEOR packages
wget -c http://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz
tar -zxvf meteor-1.5.tar.gz
mkdir ABSOLUTE/PATH/TO/federatedscope/nlp/metric/meteor/data/
mv meteor-1.5/data/paraphrase-en.gz ABSOLUTE/PATH/TO/federatedscope/nlp/metric/meteor/data/
mv meteor-1.5/meteor-1.5.jar ABSOLUTE/PATH/TO/federatedscope/nlp/metric/meteor/
```

### Reproduction
**Isolated Training**
```bash
bash run_isolated.sh $DEVICE
```

**FedAvg**
```bash
bash run_fedavg.sh $DEVICE
```

**ATC Assign training stage**
```bash
bash run_pretrain.sh $DEVICE
```

**ATC Contrast training stage**
```bash
bash run_atc.sh $DEVICE
```

### Publications
If you find this repository useful for your research or development, please cite the following [paper](https://arxiv.org/abs/2212.05789):
```
@article{dong2022collaborating,
  title = {Collaborating Heterogeneous Natural Language Processing Tasks via Federated Learning},
  author = {Dong, Chenhe and Xie, Yuexiang and Ding, Bolin and Shen, Ying and Li, Yaliang},
  journal = {arXiv preprint arXiv:2212.05789},
  year = {2022}
}
```
