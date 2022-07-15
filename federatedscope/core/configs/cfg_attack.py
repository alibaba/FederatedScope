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
    cfg.attack.label_type = 'dirty' # dirty, clean_label, dirty-label attack is all2one attack.
    cfg.attack.scale_poisoning = False
    cfg.attack.scale_para = 1.0
    cfg.attack.pgd_poisoning = False
    cfg.attack.pgd_lr = 0.1
    cfg.attack.pgd_eps = 2


    # for backdoor attack, we move the normalization into the training process. 
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


# attack_method: backdoor
# attacker_id: 0
# trigger_type: edge
# target_label_ind: 1