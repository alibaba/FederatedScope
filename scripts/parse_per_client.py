import collections
import json
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_local_training(name, path_f):
    file = open(path_f, 'r')

    acc_client = collections.defaultdict(list)
    los_client = collections.defaultdict(list)

    acc_final = collections.defaultdict(list)

    rounds = list()
    for line in file:
        if "INFO: {'Role': 'Client #" in line:
            res = json.loads(s=line.split("INFO: ")[-1].replace("\'", "\""))
            client_id = res["Role"]
            round = res["Round"]

            if round != "Final":
                # training loss curve
                acc_client[client_id].append(res["Results_raw"]["train_acc"])
                los_client[client_id].append(res["Results_raw"]["train_avg_loss"])
                if len(rounds) != 0:
                    if rounds[-1] != round:
                        rounds.append(round)
                else:
                    rounds.append(round)
            else:
                # final test_acc
                acc_final[client_id].append(res["Results_raw"]["test_acc"])

    file.close()

    idx_start = [id for id, _ in enumerate(rounds) if _ == 0]
    idx_end = idx_start[1:] + [len(rounds)]
    length_min = min([end-start for start, end in zip(idx_start, idx_end)])

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
            avg_acc += np.array(acc_client[key][start:start+length_min]) / len(idx_start)
            avg_los += np.array(los_client[key][start:start+length_min]) / len(idx_start)

        plt.subplot(121)
        plt.plot(avg_acc * 100, label=key, color=cmap(i))

        plt.subplot(122)
        plt.plot(avg_los, label=key, color=cmap(i))

    plt.legend(bbox_to_anchor=(1, 1.03))
    plt.tight_layout()

    os.makedirs("/mnt/gaodawei.gdw/FederatedScope/figure", exist_ok=True)
    plt.savefig("/mnt/gaodawei.gdw/FederatedScope/figure/exp_training_{}.pdf".format(name))
    plt.show()

    print(
        name,
        [np.mean(acc_final[k]) for k in sorted(acc_final.keys(), key=lambda x: int(x.split("#")[-1]))],
        [np.std(acc_final[k]) for k in sorted(acc_final.keys(), key=lambda x: int(x.split("#")[-1]))]
        # [k for k in sorted(acc_final.keys(), key=lambda x: int(x.split("#")[-1]))],
    )

def plot_acc_delta():
    isolated= [
        0.8596491228070176, 0.8230452674897119, 0.8297872340425533, 0.6666666666666666, 0.5410628019323672, 0.8533333333333334, 0.5308191403081914, 0.662826420890937, 0.5669895076674738, 0.6568627450980392, 0.7089201877934271, 0.985, 0.7033333333333335],[
        0.03282155602433283, 0.005819808898654738, 0.022981349994354132, 0.034950389145853564, 0.01807563954963255, 0.01896634446123506, 0.025563532055919668, 0.09087193305321604, 0.04302318161364847, 0.12264701963918434, 0.02894091081206093, 0.0, 0.036590830666833565]
    fedavg=[
        0.5526315789473685, 0.3950617283950617, 0.7553191489361702, 0.6644736842105263, 0.6231884057971014, 0.8225000000000001, 0.5004055150040552, 0.521505376344086, 0.5201775625504439, 0.6225490196078431, 0.7276995305164319, 0.985, 0.5066666666666667],[
        0.0, 0.2968102538313903, 0.0, 0.0, 0.0, 1.1102230246251565e-16, 0.001517298210370615, 0.10211877126389055, 0.0005707076522893906, 0.0069324194233975, 0.01756646660457252, 0.0, 0.015456030825826186]
    fedavg_ft=[
        0.5526315789473685, 0.8148148148148148, 0.7553191489361702, 0.4451754385964912, 0.6231884057971014, 0.7841666666666667, 0.5174371451743714, 0.5956221198156681, 0.5217917675544794, 0.6225490196078431, 0.704225352112676, 0.985, 0.555],[
        0.0, 0.0, 0.0, 0.15506727657599728, 0.0, 0.05421151989096864, 0.014943654316859897, 0.10214331887224402, 0.001976989300067154, 0.0069324194233975, 0.0, 0.0, 0.028577380332470384]
    # fedbn=[
    #     0.7456140350877193, 0.419753086419753, 0.7836879432624114, 0.668859649122807, 0.5507246376811593, 0.8799999999999999, 0.5725871857258719, 0.7089093701996928, 0.5250201775625505, 0.7058823529411765, 0.6338028169014085, 0.985, 0.5816666666666667],[
    #     0.13814048901775106, 0.27935082713542614, 0.02005976684217154, 0.00620269106303991, 0.059166418907806224, 0.023003622903070415, 0.04193503575278187, 0.02549067092976358, 0.018342725064347844, 0.07498558108224686, 0.12959787551025598, 0.0, 0.05421151989096864]
    # fedbn_ft=[
    #     0.850877192982456, 0.7448559670781894, 0.7695035460992908, 0.6140350877192983, 0.5120772946859904, 0.8641666666666667, 0.5498783454987834, 0.7265745007680492, 0.6549636803874092, 0.7254901960784313, 0.5774647887323944, 0.985, 0.6849999999999999],[
    #     0.03282155602433283, 0.10778436916632754, 0.020059766842171537, 0.06673300023992518, 0.03803868538169954, 0.007168604389202217, 0.02408636237618407, 0.004445405876647564, 0.008445701992113128, 0.030217715700828297, 0.11090151935227903, 0.0, 0.04708148963941844]
    ditto=[
        0.5526315789473685, 0.3950617283950617, 0.7553191489361702, 0.6644736842105263, 0.6231884057971014, 0.8225000000000001, 0.5178426601784266, 0.5464669738863287, 0.5209846650524617, 0.6176470588235294, 0.704225352112676, 0.985, 0.48],[
        0.0, 0.2968102538313903, 0.0, 0.0, 0.0, 1.1102230246251565e-16, 0.012339922185484943, 0.038559586378068264, 0.001141415304578781, 0.0, 0.0, 0.0, 0.007071067811865481]
    fedprox=[
        0.5526315789473685, 0.3950617283950617, 0.7553191489361702, 0.46491228070175433, 0.6231884057971014, 0.8225000000000001, 0.5024330900243309, 0.5176651305683564, 0.5447941888619855, 0.5098039215686274, 0.6478873239436619, 0.985, 0.52],[
        0.0, 0.2968102538313903, 0.0, 0.14316665564867953, 0.0, 1.1102230246251565e-16, 0.0026280375905952448, 0.09594929192244486, 0.03681396149088132, 0.07625661365737278, 0.10025425588761794, 0.0, 0.010801234497346443]
    # maml=[
    #     0.5526315789473685, 0.6131687242798354, 0.8049645390070923, 0.6951754385964911, 0.6135265700483091, 0.8191666666666667, 0.675993511759935, 0.7446236559139785, 0.6372074253430186, 0.642156862745098, 0.6150234741784038, 0.985, 0.755],[
    #     0.0, 0.2851706360340809, 0.01808162948082546, 0.034950389145853564, 0.024632944510110045, 0.023658449277630646, 0.02131173291722776, 0.004827114089852741, 0.019974767830128488, 0.0909178283872128, 0.14652100032232143, 0.0, 0.016329931618554536]

    # the following is the updated results that change "batch" into "epoch"
    maml=[
        0.8157894736842105, 0.839506172839506, 0.7553191489361702, 0.5219298245614036, 0.5555555555555556, 0.9241666666666667, 0.746147607461476, 0.8022273425499232, 0.7118644067796609, 0.6862745098039217, 0.6854460093896714, 0.985, 0.775],[
        0.0372161463782393, 0.010080204702811431, 0.0, 0.12134943273909582, 0.006831949576681597, 0.015320646925708569, 0.013522572587618043, 0.006673635636322105, 0.011139768960324625, 0.018341457778303656, 0.01756646660457247, 0.0, 0.0177951304200522]
    fedbn=[
        0.543859649122807, 0.8230452674897119, 0.7836879432624113, 0.5350877192982456, 0.5942028985507247, 0.8591666666666667, 0.7303325223033251, 0.8018433179723502, 0.7425343018563356, 0.6519607843137255, 0.5727699530516431, 0.945, 0.6433333333333333],[
        0.11833980318624597, 0.005819808898654738, 0.01808162948082546, 0.12726858007497288, 0.04266551143153549, 0.02664061227190966, 0.017498336585881104, 0.011287971164899435, 0.011916725633763876, 0.07336583111321456, 0.17828016475161995, 0.03240370349203928, 0.04938510796676352]
    fedbn_ft=[
        0.7894736842105262, 0.7695473251028807, 0.7553191489361702, 0.5109649122807017, 0.6135265700483091, 0.8816666666666667, 0.7331711273317113, 0.8026113671274961, 0.7070217917675544, 0.7107843137254902, 0.6431924882629109, 0.9816666666666666, 0.6833333333333332],[
        0.02148675212967698, 0.05073591771990927, 0.0, 0.0752675539549584, 0.05592191740478365, 0.03097938382573516, 0.008958119235350566, 0.006402969278238512, 0.042019743732833306, 0.0069324194233975, 0.08707623000465452, 0.004714045207910321, 0.02460803843372231]


    for name, (mean, std) in zip(["fedavg", "fedavg_ft", "fedbn", "fedbn_ft", "ditto", "fedprox", "maml"],[fedavg, fedavg_ft, fedbn, fedbn_ft, ditto, fedprox, maml]):
        plt.hlines(y=0, xmin=-1, xmax=15, color="black", linestyles="--")
        plt.xlim(0, 14)
        plt.xticks(ticks=[1,3,5,7,9, 11 ,13])
        plt.xlabel("Client ID")
        plt.ylabel("Accuracy Diff (%)")
        plt.bar(np.arange(len(mean))+1, (np.array(mean)-np.array(isolated[0]))*100)
        plt.savefig("/mnt/gaodawei.gdw/FederatedScope/figure/exp_acc_diff_{}.pdf".format(name))
        plt.show()

if __name__ == '__main__':
    # logs = [
    #     ("isolated", "isolated/gin_lr-0.5_step-4_pt-20_on_fs_contest_data.log"),
    #     ("fedavg", "fedavg/gin_lr-0.5_step-4_on_fs_contest_data.log"),
    #     ("fedavg_ft", "fedavg_ft/gin_lr-0.1_step-4_lstep-5_on_fs_contest_data.log"),
    #     ("fedbn", "fedbn/gin_lr-0.5_step-4_on_fs_contest_data.log"),
    #     ("fedbn_ft", "fedbn_ft/gin_lr-0.5_step-4_lstep-15_on_fs_contest_data.log"),
    #     ("ditto", "ditto/gin_lr-0.5_step-2_bs-_on_fs_contest_data.log"),
    #     # ("fedopt", "fedopt/gin_"),
    #     ("fedprox", "fedprox/gin_lr-0.5_step-4_mu-0.5_on_fs_contest_data.log"),
    #     ("maml", "maml/gin_lr-0.5_step-4_mstep-15_on_fs_contest_data.log")

    #     # update results
    #     ("maml_epoch", "maml_epoch/gin_lr-0.01_step-1_mstep-300_ilr-0.01_on_fs_contest_data.log"),
    #     ("fedbn_epoch", "fedbn_epoch/gin_lr-0.05_step-10_on_fs_contest_data.log"),
    #     ("fedbn_ft_epoch", "fedbn_ft_epoch/gin_lr-0.01_step-10_lstep-5_on_fs_contest_data.log")
    # ]
    # for name, path_f in logs:
    #     plot_local_training(name, "/mnt/gaodawei.gdw/FederatedScope/exp_out/" + path_f)

    plot_acc_delta()