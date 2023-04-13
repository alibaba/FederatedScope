import argparse
import json
import copy
import numpy as np

parser = argparse.ArgumentParser(description='FederatedScope result parsing')
parser.add_argument('--input',
                    help='path of exp results',
                    required=True,
                    type=str)
args = parser.parse_args()


def merge_local_results(local_results):
    aggr_results = copy.deepcopy(local_results[0])
    aggr_results = {key: [aggr_results[key]] for key in aggr_results}
    for i in range(1, len(local_results)):
        for k, v in local_results[i].items():
            aggr_results[k].append(v)
    return aggr_results


def main():
    result_list_wavg = []
    result_list_avg = []
    result_list_global = []

    with open(args.input, 'r') as ips:
        for line in ips:
            try:
                state, line = line.split('INFO: ')
            except:
                continue
            if line.startswith('{'):
                line = line.replace("\'", "\"")
                line = json.loads(s=line)
                if line['Round'] == 'Final' and line['Role'] == 'Server #':
                    res = line['Results_raw']
                    if 'Results_raw' in line.keys():
                        if 'server_global_eval' in res.keys():
                            result_list_global.append(
                                res['server_global_eval'])
                        if 'client_summarized_weighted_avg' in res.keys():
                            result_list_wavg.append(
                                res['client_summarized_weighted_avg'])
                        if 'client_summarized_avg' in res.keys():
                            result_list_avg.append(
                                res['client_summarized_avg'])

    print(args.input)
    if len(result_list_wavg):
        print('\tResults_weighted_avg')
        for key, v in merge_local_results(result_list_wavg).items():
            print("\t{}, {:.4f}, {:.4f}".format(key, np.mean(v), np.std(v)))
    if len(result_list_avg):
        print('\tResults_avg')
        for key, v in merge_local_results(result_list_avg).items():
            print("\t{}, {:.4f}, {:.4f}".format(key, np.mean(v), np.std(v)))
    if len(result_list_global):
        print('\tserver_global_eval')
        for key, v in merge_local_results(result_list_global).items():
            print("\t{}, {:.4f}, {:.4f}".format(key, np.mean(v), np.std(v)))


if __name__ == "__main__":
    main()
