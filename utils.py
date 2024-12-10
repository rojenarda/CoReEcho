import os
import torch
import logging
import argparse

from coreecho import get_feature_extractor
from coreecho.loss import RnCLoss
from coreecho.utils import load_model
from coreecho.regressor import get_shallow_mlp_head

print = logging.info

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    
    eta_min = lr * args.lr_decay_rate
    
    if args.lr_step_epoch != -1:
        if epoch >= args.lr_step_epoch:
            lr = eta_min
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def parse_option(custom_args=None, stage: int = 1):
    assert stage in [1, 2, 3], "Invalid stage"
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=400, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--lr_step_epoch', type=int, default=15)
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--val_n_clips_per_sample', type=int, default=1)
    parser.add_argument('--trial', type=int if stage == 1 else str, default=0 if stage == 1 else '0', help='id for recording multiple runs')
    parser.add_argument('--data_folder', type=str, default='./data', help='path to custom dataset')
    parser.add_argument('--model', type=str, default='uniformer_small', choices=['uniformer_small'])
    parser.add_argument('--aug', action='store_true', help='whether to use augmentations')
    parser.add_argument('--temp', type=float, default=2, help='temperature')
    parser.add_argument('--label_diff', type=str, default='l1', choices=['l1'], help='label distance function')
    parser.add_argument('--feature_sim', type=str, default='l2', choices=['l2'], help='feature similarity function')
    parser.add_argument('--frames', type=int)
    parser.add_argument('--frequency', type=int)
    parser.add_argument('--pretrained_weights', type=str, default=None)
    parser.add_argument('--project_name', type=str, default='coreecho')
    parser.add_argument('--model_path', type=str, default='.')
    parser.add_argument('--path_test_start_indexes', type=str, default=None)
    parser.add_argument('--path_save_test_files', type=str, default=None)
    parser.add_argument('--comet_api_key', type=str, default=None)
    if custom_args:
        opt = parser.parse_args(custom_args)
    else:
        opt = parser.parse_args()

    opt.optim = 'adamw'

    if stage == 1:
        opt.model_name = 'RnC+L1SG_{}_ep_{}_lr_{}_d_{}_wd_{}_bsz_{}_aug_{}_temp_{}_label_{}_feature_{}_trial_{}'. \
            format(
                opt.model, opt.epochs, opt.learning_rate, opt.lr_decay_rate, opt.weight_decay,
                opt.batch_size, opt.aug, opt.temp, opt.label_diff, opt.feature_sim, opt.trial
            )
    elif stage == 2 or stage == 3:
        opt.model_name = 'RnC (LP)_{}_ep_{}__d_{}_wd_{}_bsz_{}_aug_{}_temp_{}_label_{}_feature_{}_trial_{}'. \
            format(
                opt.model, opt.epochs, opt.learning_rate, opt.weight_decay, opt.batch_size,
                opt.aug, opt.temp, opt.label_diff, opt.feature_sim, opt.trial
            )
    else:
        raise ValueError("Invalid stage")

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    else:
        print('WARNING: folder exist.')

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(opt.save_folder, 'training.log')),
            logging.StreamHandler()
        ])

    print(f"Model name: {opt.model_name}")
    print(f"Options: {opt}")

    return opt

def set_model(opt, stage: int = 1):
    assert stage in [1, 2, 3], "Invalid stage"
    model = get_feature_extractor(opt.model, opt.pretrained_weights if stage == 1 else None)
    if opt.model == 'uniformer_small':
        dim_in = model.head.in_features
    else:
        dim_in = model.fc.in_features
    dim_out = 1

    regressor = get_shallow_mlp_head(dim_in, dim_out)

    if stage == 1:
        criterion = RnCLoss(temperature=opt.temp, label_diff=opt.label_diff, feature_sim=opt.feature_sim)
    else:
        checkpoint = torch.load(opt.pretrained_weights, map_location='cpu')
        model = load_model(model, checkpoint['model'])
        regressor = load_model(regressor, checkpoint['regressor'])


    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            regressor = torch.nn.DataParallel(regressor)
        model = model.cuda()
        regressor = regressor.cuda()
        if stage == 1:
            criterion = criterion.cuda()
        torch.backends.cudnn.benchmark = True

    if stage == 1:
        return model, criterion, regressor
    else:
        return model, regressor