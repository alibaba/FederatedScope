set -e

# Graph
conda install -y pyg==2.0.4 -c pyg
conda install -y rdkit=2021.09.4=py39hccf6a74_0 -c conda-forge
conda install -y nltk

# Speech and NLP
conda install -y sentencepiece textgrid typeguard -c conda-forge
conda install -y transformers==4.16.2 tokenizers==0.10.3 datasets -c huggingface -c conda-forge
conda install -y torchtext -c pytorch

conda clean -a -y