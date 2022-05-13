import collections
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_local_training(name, path_f):
    file = open(path_f, 'r')

    acc_client = collections.defaultdict(list)
    los_client = collections.defaultdict(list)

    rounds = list()
    for line in file:
        if "INFO: {'Role': 'Client #" in line:
            res = json.loads(s=line.split("INFO: ")[-1].replace("\'", "\""))
            client_id = res["Role"]
            acc_client[client_id].append(res["Results_raw"]["train_acc"])
            los_client[client_id].append(res["Results_raw"]["train_avg_loss"])

            round = res["Round"]
            if len(rounds) != 0:
                if rounds[-1] != round:
                    rounds.append(round)
            else:
                rounds.append(round)

    file.close()

    idx_start = [id for id, _ in enumerate(rounds) if _ == 0]
    idx_end = idx_start[1:] + [len(rounds)]

    # single
    # for start, end in zip(idx_start, idx_end):
    #     plt.figure(figsize=[12, 6])
    #     plt.subplot(1, 2, 1)
    #     plt.title("accuracy")
    #     plt.subplot(1, 2, 2)
    #     plt.title("average loss")
    #
    #     for key in acc_client.keys():
    #         plt.subplot(121)
    #         plt.plot(acc_client[key][start:end], label=key)
    #
    #         plt.subplot(122)
    #         plt.plot(los_client[key][start:end], label=key)
    #
    #     plt.legend()
    #     plt.show()

    # avg
    ax = plt.figure(figsize=[12, 3.5])
    plt.subplot(1, 2, 1)
    plt.xlabel("Federated training rounds")
    plt.ylabel("Accuracy(\%)")
    plt.subplot(1, 2, 2)
    plt.xlabel("Federated training rounds")
    plt.ylabel("Loss")

    cmap = plt.get_cmap('tab20')
    for i, key in enumerate(sorted(acc_client.keys(), key=lambda x: int(x.split('#')[-1]))):
        avg_acc, avg_los = 0., 0.
        for start, end in zip(idx_start, idx_end):
            avg_acc += np.array(acc_client[key][start:end]) / len(idx_start)
            avg_los += np.array(los_client[key][start:end]) / len(idx_start)

        plt.subplot(121)
        plt.plot(avg_acc * 100, label=key, color=cmap(i))

        plt.subplot(122)
        plt.plot(avg_los, label=key, color=cmap(i))


    plt.legend(bbox_to_anchor=(1, 1.03))
    plt.tight_layout()

    os.makedirs("/mnt/gaodawei.gdw/FederatedScope/figure", exist_ok=True)
    plt.savefig("/mnt/gaodawei.gdw/FederatedScope/figure/exp_training_{}.pdf".format(name))
    plt.show()

if __name__ == '__main__':
    logs = [
        ('fedavg', "/mnt/gaodawei.gdw/FederatedScope/out_fedavg/gin_0.5_2_on_fs_contest_data_ooxx.log"),
        ('isolated', "")
        ('fedbn', ""),
        ('ditto', ""),
        ('maml', "")
    ]
    for name, path_f in logs:
        plot_local_training(name, path_f)