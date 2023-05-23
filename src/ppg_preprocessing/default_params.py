# ---------------------------------------------------------------------------------------------------------------
#                                Default Params
# ---------------------------------------------------------------------------------------------------------------
default_params = {}
default_params['project'] = {'name': 'default_project', 'disable_comet': False}
default_params['dataset'] = {'sleep_encoding': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},  # {0: 0, 1: 1, 2: 1, 3: 2, 4: 3},
                             'epoch_duration': 30,
                             'max_epochs': 1200,
                             'filter_type': 'cheby_lowpass',
                             'ripple': 0.001,
                             'filter_highcut': 8,
                             'filter_order': 8,
                             'resample_fs': 100  # 1024 / 30,

                             }
default_params['dataset']['samples_per_epoch'] = int(
    default_params['dataset']['epoch_duration'] * default_params['dataset']['resample_fs'])
default_params['dataset']['max_samples'] = int(
    default_params['dataset']['samples_per_epoch'] * default_params['dataset']['max_epochs'])
default_params['dataset']['win_len'] = default_params['dataset']['samples_per_epoch']
default_params['dataset']['epochs_overlap_per_sequence'] = 0
default_params['dataset']['train_X_mean'] = -0.00011141635
default_params['dataset']['train_X_std'] = 0.017052623

import copy


def specify_dataset(params, dataset_name):
    dataset_params = copy.deepcopy(params)
    dataset_params['dataset']['name'] = dataset_name
    return dataset_params
