import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging
import os
import numpy as np
import federatedscope.register as register

logger = logging.getLogger(__name__)


def label_to_onehot(target, num_classes=100):
    return torch.nn.functional.one_hot(target, num_classes)


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))


def iDLG_trick(original_gradient, num_class, is_one_hot_label=False):
    '''
    Using iDLG trick to recover the label. Paper: "iDLG: Improved Deep
    Leakage from Gradients", link: https://arxiv.org/abs/2001.02610

    Args:
        original_gradient: the gradient of the FL model; type: list
        num_class: the total number of class in the data
        is_one_hot_label: whether the dataset's label is in the form of one
        hot. Type: bool

    Returns:
        The recovered label by iDLG trick.

    '''
    last_weight_min = torch.argmin(torch.sum(original_gradient[-2], dim=-1),
                                   dim=-1).detach()

    if is_one_hot_label:
        label = label_to_onehot(
            last_weight_min.reshape((1, )).requires_grad_(False), num_class)
    else:
        label = last_weight_min
    return label


def cos_sim(input_gradient, gt_gradient):
    total = 1 - torch.nn.functional.cosine_similarity(
        input_gradient.flatten(), gt_gradient.flatten(), 0, 1e-10)

    # total = 0
    # input_norm= 0
    # gt_norm = 0
    #
    # total -= (input_gradient * gt_gradient).sum()
    # input_norm += input_gradient.pow(2).sum()
    # gt_norm += gt_gradient.pow(2).sum()
    # total += 1 + total / input_norm.sqrt() / gt_norm.sqrt()

    return total


def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

    total = x.size()[0]
    for ind in range(1, len(x.size())):
        total *= x.size()[ind]
    return (dx + dy) / (total)


def approximate_func(x, device, C1=20, C2=0.5):
    '''
    Approximate the function f(x) = 0 if x<0.5 otherwise 1
    Args:
        x: input data;
        device:
        C1:
        C2:

    Returns:
        1/(1+e^{-1*C1 (x-C2)})

    '''
    C1 = torch.tensor(C1).to(torch.device(device))
    C2 = torch.tensor(C2).to(torch.device(device))

    return 1 / (1 + torch.exp(-1 * C1 * (x - C2)))


def get_classifier(classifier: str, model=None):
    if model is not None:
        return model

    if classifier == 'lr':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=0)
        return model
    elif classifier.lower() == 'randomforest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=0)
        return model
    elif classifier.lower() == 'svm':
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        return model
    else:
        ValueError()


def get_data_info(dataset_name):
    '''
    Get the dataset information, including the feature dimension, number of
    total classes, whether the label is represented in one-hot version

    Args:
        dataset_name:dataset name; str

    :returns:
        data_feature_dim, num_class, is_one_hot_label

    '''
    if dataset_name.lower() == 'femnist':

        return [1, 28, 28], 36, False
    else:
        ValueError(
            'Please provide the data info of {}: data_feature_dim, num_class'.
            format(dataset_name))


def get_data_sav_fn(dataset_name):
    if dataset_name.lower() == 'femnist':
        return sav_femnist_image
    else:
        logger.info(f"Reconstructed data saving function is not provided for "
                    f"dataset: {dataset_name}")
        return None


def sav_femnist_image(data, sav_pth, name):

    _ = plt.figure(figsize=(4, 4))
    # print(data.shape)

    if len(data.shape) == 2:
        data = torch.unsqueeze(data, 0)
        data = torch.unsqueeze(data, 0)

    ind = min(data.shape[0], 16)
    # print(data.shape)

    # plt.imshow(data * 127.5 + 127.5, cmap='gray')

    for i in range(ind):
        plt.subplot(4, 4, i + 1)

        plt.imshow(data[i, 0, :, :] * 127.5 + 127.5, cmap='gray')
        # plt.imshow(generated_data[i, 0, :, :] , cmap='gray')
        # plt.imshow()
        plt.axis('off')

    plt.savefig(os.path.join(sav_pth, name))
    plt.close()


def get_info_diff_loss(info_diff_type):
    if info_diff_type.lower() == 'l2':
        info_diff_loss = torch.nn.MSELoss(reduction='sum')
    elif info_diff_type.lower() == 'l1':
        info_diff_loss = torch.nn.SmoothL1Loss(reduction='sum', beta=1e-5)
    elif info_diff_type.lower() == 'sim':
        info_diff_loss = cos_sim
    else:
        ValueError(
            'info_diff_type: {} is not supported'.format(info_diff_type))
    return info_diff_loss


def get_reconstructor(atk_method, **kwargs):
    '''

    Args:
        atk_method: the attack method name, and currently supporting "DLG:
        deep leakage from gradient", and "IG: Inverting gradient" ; Type: str
        **kwargs: other arguments

    Returns:

    '''

    if atk_method.lower() == 'dlg':
        from federatedscope.attack.privacy_attacks.reconstruction_opt import\
            DLG
        logger.info(
            '--------- Getting reconstructor: DLG --------------------')

        return DLG(max_ite=kwargs['max_ite'],
                   lr=kwargs['lr'],
                   federate_loss_fn=kwargs['federate_loss_fn'],
                   device=kwargs['device'],
                   federate_lr=kwargs['federate_lr'],
                   optim=kwargs['optim'],
                   info_diff_type=kwargs['info_diff_type'],
                   federate_method=kwargs['federate_method'])
    elif atk_method.lower() == 'ig':
        from federatedscope.attack.privacy_attacks.reconstruction_opt import\
            InvertGradient
        logger.info(
            '------- Getting reconstructor: InvertGradient ------------------')
        return InvertGradient(max_ite=kwargs['max_ite'],
                              lr=kwargs['lr'],
                              federate_loss_fn=kwargs['federate_loss_fn'],
                              device=kwargs['device'],
                              federate_lr=kwargs['federate_lr'],
                              optim=kwargs['optim'],
                              info_diff_type=kwargs['info_diff_type'],
                              federate_method=kwargs['federate_method'],
                              alpha_TV=kwargs['alpha_TV'])
    else:
        ValueError(
            "attack method: {} lacks reconstructor implementation".format(
                atk_method))


def get_generator(dataset_name):
    '''
    Get the dataset's corresponding generator.
    Args:
        dataset_name: The dataset name; Type: str

    :returns:
        The generator; Type: object

    '''
    if dataset_name == 'femnist':
        from federatedscope.attack.models.gan_based_model import \
            GeneratorFemnist
        return GeneratorFemnist
    else:
        ValueError(
            "The generator to generate data like {} is not defined!".format(
                dataset_name))


def get_data_property(ctx):
    # A SHOWCASE for Femnist dataset: Property := whether contains a circle.
    x, label = [_.to(ctx.device) for _ in ctx.data_batch]

    prop = torch.zeros(label.size)
    positive_labels = [0, 6, 8]
    for ind in range(label.size()[0]):
        if label[ind] in positive_labels:
            prop[ind] = 1
    prop.to(ctx.device)
    return prop


def get_passive_PIA_auxiliary_dataset(dataset_name):
    '''

    Args:
        dataset_name (str): dataset name

    :returns:

    the auxiliary dataset for property inference attack. Type: dict

    {
        'x': array,
        'y': array,
        'prop': array
                    }

    '''
    for func in register.auxiliary_data_loader_PIA_dict.values():
        criterion = func(dataset_name)
        if criterion is not None:
            return criterion
    if dataset_name == 'toy':

        def _generate_data(instance_num=1000, feature_num=5, save_data=False):
            """
            Generate data in FedRunner format
            Args:
                instance_num:
                feature_num:
                save_data:

            Returns:
                {
                            'x': ...,
                            'y': ...,
                            'prop': ...
                        }

            """
            weights = np.random.normal(loc=0.0, scale=1.0, size=feature_num)
            bias = np.random.normal(loc=0.0, scale=1.0)

            prop_weights = np.random.normal(loc=0.0,
                                            scale=1.0,
                                            size=feature_num)
            prop_bias = np.random.normal(loc=0.0, scale=1.0)

            x = np.random.normal(loc=0.0,
                                 scale=0.5,
                                 size=(instance_num, feature_num))
            y = np.sum(x * weights, axis=-1) + bias
            y = np.expand_dims(y, -1)
            prop = np.sum(x * prop_weights, axis=-1) + prop_bias
            prop = 1.0 * ((1 / (1 + np.exp(-1 * prop))) > 0.5)
            prop = np.expand_dims(prop, -1)

            data_train = {'x': x, 'y': y, 'prop': prop}
            return data_train

        return _generate_data()
    else:
        ValueError(
            'The data cannot be loaded. Please specify the data load function.'
        )
