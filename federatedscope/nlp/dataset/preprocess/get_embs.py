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
lines = [l.split() for l in lines]
vocab = [l[0] for l in lines]
emb_floats = [[float(n) for n in l[1:]] for l in lines]
emb_floats.append([0.0 for _ in range(300)])  # for unknown word
js = {'vocab': vocab, 'emba': emb_floats}
with open('embs.json', 'w') as ouf:
    json.dump(js, ouf)
