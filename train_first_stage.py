import argparse
import os
import sys
import logging
import torch
import time

from coreecho.validation import validate
from coreecho.dataloader import set_loader
from coreecho.viz import HelperTSNE, HelperUMAP
from coreecho.utils import AverageMeter, save_model, set_seed, set_optimizer

from comet_ml import Experiment

from .utils import parse_option, set_model, adjust_learning_rate

print = logging.info

def train(train_loader, model, criterion, optimizer, epoch, opt, regressor, optimizer_regressor):
    model.train()
    regressor.train()

    criterion_mse = torch.nn.L1Loss()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)

        views1, views2 = batch
        images = torch.cat([views1["image"], views2["image"]], dim=0)
        labels = views1["label"]
        bsz = labels.shape[0]

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        _, features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = criterion(features, labels)
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        features = features.detach()
        y_preds = regressor(torch.cat((features[:,0], features[:,1]), dim=0))
        loss_reg = criterion_mse(y_preds, labels.repeat(2, 1))

        optimizer_regressor.zero_grad()
        loss_reg.backward()
        optimizer_regressor.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            to_print = 'Train: [{0}][{1}/{2}]\t' \
                        'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'DT {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                        'loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses
            )
            print(to_print)
            sys.stdout.flush()

def main(stage_1_args: dict = None):
    opt = parse_option(stage_1_args)


    # Initialize Comet Experiment
    experiment = Experiment(
        api_key=opt.comet_api_key,
        project_name=opt.project_name
    )

    # Set experiment parameters
    experiment.set_name(opt.model_name)
    experiment.log_parameters(vars(opt))

    # Set seed (for reproducibility)
    set_seed(opt.trial)

    # build data loader
    train_loader, train_no_aug_loader, val_loader, test_loader = set_loader(opt)

    # build model and criterion
    model, criterion, regressor = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)
    optimizer_regressor = set_optimizer(opt, regressor)

    start_epoch = 0

    # training routine
    best_error = 1e5
    save_file_best = os.path.join(opt.save_folder, 'best.pth')
    for epoch in range(start_epoch, opt.epochs + 1):
        lr_cur_val = adjust_learning_rate(opt, optimizer, epoch)
        _ = adjust_learning_rate(opt, optimizer_regressor, epoch)

        train(train_loader, model, criterion, optimizer, epoch, opt, regressor, optimizer_regressor)

        valid_metrics, valid_aux  = validate(val_loader, model, regressor, opt.val_n_clips_per_sample)
        valid_tsne = HelperTSNE(valid_aux['embeddings'], n_components=2, perplexity=5, random_state=7)
        valid_umap = HelperUMAP(valid_aux['embeddings'], n_components=2, n_neighbors=5, init='random', random_state=0)

        valid_error = valid_metrics['l1']
        is_best = valid_error <= best_error
        best_error = min(valid_error, best_error)
        print(f"Best MAE: {best_error:.3f}")

        if is_best:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'regressor': regressor.state_dict(),
                'best_error': best_error,
            }, save_file_best)

        experiment.log_metric("epoch", epoch)
        experiment.log_metric("learning_rate", lr_cur_val)
        experiment.log_metric("Val R2", valid_metrics['r2'].item())
        experiment.log_metric("Val L2", valid_metrics['l2'].item())
        experiment.log_metric("Val L1", valid_metrics['l1'].item())
        for key, val in valid_aux['aux'].items():
            experiment.log_figure(f"Val UMAP ({key})", valid_umap(val))
            experiment.log_figure(f"Val TSNE ({key})", valid_tsne(val))

        save_file = os.path.join(opt.save_folder, 'last.pth')
        save_model(model, regressor, opt, epoch, save_file, best_error)


    experiment.end()

    print("=" * 120)
    print("Test best model on test set...")
    checkpoint = torch.load(save_file_best)
    model.load_state_dict(checkpoint['model'])
    regressor.load_state_dict(checkpoint['regressor'])
    print(f"Loaded best model, epoch {checkpoint['epoch']}, best val error {checkpoint['best_error']:.3f}")

    set_seed(opt.trial)
    test_metrics, test_aux = validate(test_loader, model, regressor, opt.val_n_clips_per_sample)
    print('Test R2: {:.3f}'.format(test_metrics['r2']))
    print('Test L2: {:.3f}'.format(test_metrics['l2']))
    print('Test L1: {:.3f}'.format(test_metrics['l1']))

    set_seed(opt.trial)
    val_metrics, val_aux = validate(val_loader, model, regressor, opt.val_n_clips_per_sample)
    print('Val R2: {:.3f}'.format(val_metrics['r2']))
    print('Val L2: {:.3f}'.format(val_metrics['l2']))
    print('Val L1: {:.3f}'.format(val_metrics['l1']))

    set_seed(opt.trial)
    train_metrics, train_aux = validate(train_no_aug_loader, model, regressor)
    print('Train R2: {:.3f}'.format(train_metrics['r2']))
    print('Train L2: {:.3f}'.format(train_metrics['l2']))
    print('Train L1: {:.3f}'.format(train_metrics['l1']))
    
    return save_file_best


if __name__ == '__main__':
    main()