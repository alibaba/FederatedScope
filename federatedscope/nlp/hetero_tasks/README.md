# 1. Environment
```bash
pip install nltk
pip install transformers

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
```

# 2. Run \textsc{Isolated} baseline
```bash
bash run_isolated.sh
```

# 3. Run \textsc{FedAvg} baseline
```bash
bash run_fednlp.sh
```

# 4. Run \textsc{Assign} stage
```bash
bash run_pfednlp_pretrain.sh
```

# 5. Run \textsc{Contrast} stage
```bash
bash run_pcfednlp.sh
```

# 6. Run \textsc{Contrast} w/o CL
```bash
bash run_pfednlp.sh
```
