import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument(
    '-f',
    help='path to .txt file containing word embedding information;',
    type=str,
    default='glove.6B.300d.txt')

args = parser.parse_args()

lines = []
with open(args.f, 'r') as inf:
    lines = inf.readlines()
lines = [i.split() for i in lines]
vocab = [i[0] for i in lines]
emb_floats = [[float(n) for n in i[1:]] for i in lines]
emb_floats.append([0.0 for _ in range(300)])  # for unknown word
js = {'vocab': vocab, 'emba': emb_floats}
with open('embs.json', 'w') as ouf:
    json.dump(js, ouf)
