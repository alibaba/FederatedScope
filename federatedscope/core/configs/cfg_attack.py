from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_attack_cfg(cfg):

    # ------------------------------------------------------------------------ #
    # attack
    # ------------------------------------------------------------------------ #
    cfg.attack = CN()
    cfg.attack.attack_method = ''
    # for gan_attack and backdoor attack
    cfg.attack.target_label_ind = -1
    cfg.attack.attacker_id = -1

    # for backdoor attack

    cfg.attack.setting = 'fix'
    cfg.attack.freq = 10
    cfg.attack.insert_round = 100000
    cfg.attack.mean = [0.1307]
    cfg.attack.std = [0.3081]
    cfg.attack.trigger_type = 'edge'
    cfg.attack.label_type = 'dirty'
    cfg.attack.edge_num = 100
    cfg.attack.poison_ratio = 0.5
    cfg.attack.scale_poisoning = False
    cfg.attack.scale_para = 1.0
    cfg.attack.pgd_poisoning = False
    cfg.attack.pgd_lr = 0.1
    cfg.attack.pgd_eps = 2
    cfg.attack.self_opt = False
    cfg.attack.self_lr = 0.05
    cfg.attack.self_epoch = 6

    # defense:

    cfg.attack.norm_clip = False
    cfg.attack.norm_clip_value = 5.0
    cfg.attack.dp_noise = -1.0
    cfg.attack.krum = False
    cfg.attack.multi_krum = False

    # Note: the mean and std should be the list type.

    # for reconstruct_opt
    cfg.attack.reconstruct_lr = 0.01
    cfg.attack.reconstruct_optim = 'Adam'
    cfg.attack.info_diff_type = 'l2'
    cfg.attack.max_ite = 400
    cfg.attack.alpha_TV = 0.001

    # for active PIA attack
    cfg.attack.alpha_prop_loss = 0

    # for passive PIA attack
    cfg.attack.classifier_PIA = 'randomforest'

    # for gradient Ascent --- MIA attack
    cfg.attack.inject_round = 0

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_attack_cfg)


def assert_attack_cfg(cfg):
    pass


register_config("attack", extend_attack_cfg)
