from torch import nn
from torch.nn.utils import spectral_norm

from federatedscope.autotune.utils import arm2dict


class EncNet(nn.Module):
    def __init__(self, in_channel, out_channel, hid_dim=64):
        super(EncNet, self).__init__()

        self.fc_layer = nn.Sequential(
            spectral_norm(nn.Linear(in_channel, hid_dim, bias=False)),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Linear(hid_dim, out_channel, bias=False)),
            nn.ReLU(inplace=True),
        )

    def forward(self, client_enc):
        mean_update = self.fc_layer(client_enc)
        return mean_update


class HyperNet(nn.Module):
    def __init__(
        self,
        input_dim,
        sizes,
        n_clients,
        device,
    ):
        super(HyperNet, self).__init__()
        self.EncNet = EncNet(input_dim, 32)
        self.out = nn.ModuleList()
        for num_cate in sizes:
            self.out.append(
                nn.Sequential(nn.Linear(32, num_cate, bias=True),
                              nn.Softmax()))

    def forward(self, encoding):
        client_enc = self.EncNet(encoding)
        probs = []
        for module in self.out:
            out = module(client_enc)
            probs.append(out)
        return probs


if __name__ == "__main__":
    import yaml
    import argparse
    import torch

    parser = argparse.ArgumentParser(description='Interpret learned policy')
    parser.add_argument('--ss_path', type=str, default='')
    parser.add_argument('--log_path', type=str, default='')
    parser.add_argument('--pt_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    args = parser.parse_args()

    with open(args.ss_path, 'r') as ips:
        arms = yaml.load(ips, Loader=yaml.FullLoader)
        print(arms)
    with open(args.log_path, 'r') as ips:
        ckpt = yaml.load(ips, Loader=yaml.FullLoader)
    stop_exploration = ckpt['stop']
    print("stop: {}".format(stop_exploration))

    psn_pi = torch.load(args.pt_path, map_location='cpu')
    client_encodings = psn_pi['client_encodings']
    policy_net = HyperNet(
        client_encodings.shape[-1],
        [len(arms)],
        client_encodings.shape[0],
        'cpu',
    ).to('cpu')
    policy_net.load_state_dict(psn_pi['policy_net'])
    policy_net.eval()
    prbs = policy_net(client_encodings)
    prbs = prbs[0].detach().numpy()
    clientwise_configs = dict()
    for i in range(prbs.shape[0]):
        arm_idx = prbs[i].argmax()
        clientwise_configs['client_{}'.format(i + 1)] = arm2dict(
            arms['arm{}'.format(arm_idx)])
    with open(args.save_path, 'w') as ops:
        yaml.Dumper.ignore_aliases = lambda *args: True
        yaml.dump(clientwise_configs, ops)
