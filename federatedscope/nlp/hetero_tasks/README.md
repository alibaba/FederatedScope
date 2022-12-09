# 1. Environment
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
mv meteor-1.5/data/paraphrase-en.gz ABSOLUTE/PATH/TO/federatedscope/nlp/hetero_tasks/metric/generation/meteor/data/
mv meteor-1.5/meteor-1.5.jar ABSOLUTE/PATH/TO/federatedscope/nlp/hetero_tasks/metric/generation/meteor/
```

# 2. Run \textsc{Isolated} baseline
```bash
bash run_isolated.sh $DEVICE
```

# 3. Run \textsc{FedAvg} baseline
```bash
bash run_fedavg.sh $DEVICE
```

# 4. Run ATC \textsc{Assign} stage
```bash
bash run_pretrain.sh $DEVICE
```

# 5. Run ATC \textsc{Contrast} stage
```bash
bash run_atc.sh $DEVICE
```
