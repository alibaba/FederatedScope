import argparse
import json
import copy
import numpy as np

parser = argparse.ArgumentParser(description='FederatedScope result parsing')
parser.add_argument('--input',
                    help='path of exp results',
                    required=True,
                    type=str)
parser.add_argument('--round',
                    help='path of exp results',
                    required=True,
                    type=int)
args = parser.parse_args()


def merge_local_results(local_results):
    aggr_results = copy.deepcopy(local_results[0])
    aggr_results = {key: [aggr_results[key]] for key in aggr_results}
    for i in range(1, len(local_results)):
        for k, v in local_results[i].items():
            aggr_results[k].append(v)
    return aggr_results


def main():
    parse(args.input, args.round)


def parse(input, round):
    result_list_wavg = []
    with open(input, 'r') as ips:
        for line in ips:
            try:
                state, line = line.split('INFO: ')
            except:
                continue
            if line.startswith('{'):
                line = line.replace("\'", "\"")
                line = json.loads(s=line)
                if line['Round'] == round and line['Role'] == 'Server #':
                    result_list_wavg.append(line["Results_weighted_avg"])

    print(args.input)
    if len(result_list_wavg):
        for key, v in merge_local_results(result_list_wavg).items():
            print("\t{}, {:.4f}, {:.4f}".format(key, np.mean(v), np.std(v)))


if __name__ == "__main__":
    main()
