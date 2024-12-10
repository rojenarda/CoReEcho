import os
import copy
import torch
import pickle
import pandas as pd

from torcheval.metrics.functional import r2_score

from coreecho.dataset import EchoNetTest
from coreecho.validation import validate

from comet_ml import Experiment

from utils import parse_option, set_model

def set_test_loader(opt):
    test_ds = EchoNetTest(
            root=opt.data_folder,
            frames=opt.frames,
            frequency=opt.frequency,
            path_test_start_indexes=opt.path_test_start_indexes,
            trial=opt.trial,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=opt.num_workers,
    )
    
    return test_loader

def main(custom_args: dict = None):
    opt = parse_option(custom_args, stage=3)

    if not os.path.exists(opt.path_save_test_files):
        os.makedirs(opt.path_save_test_files)

    experiment = Experiment(
        api_key=opt.comet_api_key,
        project_name=opt.project_name,
    )

    # Set experiment parameters
    experiment.set_name(opt.model_name)
    experiment.log_parameters(vars(opt))

    model, regressor = set_model(opt, stage=3)

    df = pd.read_pickle(opt.path_test_start_indexes)
    list_trial = list(range(len(df[list(df.keys())[0]])))

    list_outputs = []
    best_r2 = -1_000_000
    for trial in list_trial:
        opt.trial = trial
        test_loader = set_test_loader(opt)
        test_metrics, test_aux = validate(test_loader, model, regressor)
        if best_r2 <= test_metrics['r2']:
            best_r2 = max(best_r2, test_metrics['r2'])
            best_metrics = copy.deepcopy(test_metrics)
            best_aux = copy.deepcopy(test_aux)
        list_outputs.append(test_aux['outputs'])

        print('-'*10)
        print('Trial ', trial)
        print(test_metrics)
        print('')

    outputs = torch.cat(list_outputs, dim=1).mean(dim=1)[:,None]
    labels = test_aux['labels']

    metrics = {
        'r2': r2_score(outputs, labels),
        'l1': torch.nn.L1Loss()(outputs, labels),
        'l2': torch.sqrt(torch.nn.MSELoss()(outputs, labels)),
    }

    experiment.log_metrics(metrics)


    print('-'*30)
    print(f'Metrics from {len(list_trial)}x clips')
    print(metrics)

    dict_test_files = {
        'N clips': len(list_trial),
        'metrics xN clips': metrics,
        'best_metrics x1 clip': best_metrics,
        'best_aux x1 clip': best_aux,
    }

    with open(opt.path_save_test_files, 'wb') as f:
        pickle.dump(dict_test_files, f)

