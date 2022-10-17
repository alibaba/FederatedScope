import numpy as np
import pandas as pd


def load_xgb_base_data(config=None, generate=False):
    """
    To generate the synthetic data for vertical FL

    Arguments:
        config: configuration
        generate (bool): whether to generate the synthetic data
    :returns: The synthetic data, the modified config
    :rtype: dict
    """

    if generate:
        # generate data for running an hep_xgb FL example

        data = pd.read_csv("dataloader/cs-training.csv")
        # data = data.fillna(0)
        data = data[[
            'SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines', 'age',
            'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
            'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
            'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
            'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
        ]].values  # len(data) = 15000  len(data[0]) = 11
        # data = pd.DataFrame(data)
        # for i in range(0, len(data)):
        #    data[i] = data[i].fillna(data[i][1:].mean())
        # data = np.array(data)
        # subsample
        sample_size = 15000

        def balance_sample(sample_size, y):
            y_ones_idx = (y == 1).nonzero()[0]
            y_ones_idx = np.random.choice(y_ones_idx,
                                          size=int(sample_size / 2))
            y_zeros_idx = (y == 0).nonzero()[0]
            y_zeros_idx = np.random.choice(y_zeros_idx,
                                           size=int(sample_size / 2))

            y_index = np.concatenate([y_zeros_idx, y_ones_idx], axis=0)
            np.random.shuffle(y_index)
            return y_index

        sample_idx = balance_sample(sample_size, data[:, 0])
        data = data[sample_idx]
        # print('Data shape: {}'.format(data.shape))

        train_num = int(0.9 * len(data))
        # train_y, test_y = data[:train_num, 0], data[train_num:, 0]
        # train_x, test_x = data[:train_num, 1:], data[train_num:, 1:]
        y = data[:, 0]
        x = data[:, 1:]

        test_data = {'x': x[train_num:], 'y': y[train_num:]}
        data = dict()

        # For Server #0
        data[0] = dict()
        data[0]['train'] = None
        data[0]['val'] = None
        data[0]['test'] = test_data

        # For Client #1
        data[1] = dict()
        data[1]['train'] = {'x': x[:train_num, :config.xgb_base.dims[0]]}
        data[1]['val'] = None
        data[1]['test'] = test_data

        # For Client #2
        data[2] = dict()
        data[2]['train'] = {
            'x': x[:train_num, config.xgb_base.dims[0]:config.xgb_base.dims[1]]
        }
        data[2]['val'] = None
        data[2]['test'] = test_data

        # For Client #3

        data[3] = dict()
        data[3]['train'] = {
            'x': x[:train_num,
                   config.xgb_base.dims[1]:config.xgb_base.dims[2]],
            'y': y[:train_num]
        }
        data[3]['val'] = None
        data[3]['test'] = test_data

        return data, config
    else:
        raise ValueError('You must provide the data file')
