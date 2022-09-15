from dataset import load_data
from models import MRnet
from config import config
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import _train_model, _evaluate_model, _get_lr
import time
import torch.utils.data as data
import os

"""Performs training of a specified model.
    
Input params:
    config_file: Takes in configurations to train with 
"""

def train(config : dict):
    """
    Function where actual training takes place

    Args:
        config (dict) : Configuration to train with
    """
    
    print('Starting to Train Model...')

    train_loader, val_loader, train_wts, val_wts = load_data(config['task'])

    print('Initializing Model...')
    model = MRnet()
    if torch.cuda.is_available():
        model = model.cuda()
        train_wts = train_wts.cuda()
        val_wts = val_wts.cuda()

    print('Initializing Loss Method...')
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=train_wts)
    val_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=val_wts)

    if torch.cuda.is_available():
        criterion = criterion.cuda()
        val_criterion = val_criterion.cuda()

    print('Setup the Optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=.3, threshold=1e-4, verbose=True)
    
    starting_epoch = config['starting_epoch']
    num_epochs = config['max_epoch']
    patience = config['patience']
    log_train = config['log_train']
    log_val = config['log_val']

    best_val_loss = float('inf')
    best_val_auc = float(0)

    print('Starting Training')

    writer = SummaryWriter(comment='lr={} task={}'.format(config['lr'], config['task']))
    t_start_training = time.time()

    for epoch in range(starting_epoch, num_epochs):

        current_lr = _get_lr(optimizer)
        epoch_start_time = time.time()  # timer for entire epoch

        print('Started Training')
        train_loss, train_auc = _train_model(
            model, train_loader, epoch, num_epochs, optimizer, criterion, writer, current_lr, log_every=log_train)

        print('train loop ended, now val')
        val_loss, val_auc = _evaluate_model(
            model, val_loader, val_criterion,  epoch, num_epochs, writer, current_lr, log_val)

        writer.add_scalar('Train/Avg Loss', train_loss, epoch)
        writer.add_scalar('Val/Avg Loss', val_loss, epoch)

        scheduler.step(val_loss)

        t_end = time.time()
        delta = t_end - epoch_start_time

        print("train loss : {0} | train auc {1} | val loss {2} | val auc {3} | elapsed time {4} s".format(
            train_loss, train_auc, val_loss, val_auc, delta))

        print('-' * 30)

        writer.flush()

        if val_auc > best_val_auc:
            best_val_auc = val_auc

        if bool(config['save_model']):
            file_name = 'model_{}_{}_val_auc_{:0.4f}_train_auc_{:0.4f}_epoch_{}.pth'.format(config['exp_name'], config['task'], val_auc, train_auc, epoch+1)
            torch.save({
                'model_state_dict': model.state_dict()
            }, './weights/{}/{}'.format(config['task'],file_name))

    t_end_training = time.time()
    print(f'training took {t_end_training - t_start_training} s')
    writer.flush()
    writer.close()

if __name__ == '__main__':

    print('Training Configuration')
    print(config)

    train(config=config)

    print('Training Ended...')
