import getopt
import sys
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from .lstm import ActionClassificationLSTM, PoseDataModule, WINDOW_SIZE


def configuration_parser(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    # training batch size
    parser.add_argument('--batch_size', type=int, default=512)
    # max training epochs = 400
    parser.add_argument('--epochs', type=int, default=400)
    # training initial learning rate
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    # number of classes = number of human actions in the data set= 6
    parser.add_argument('--num_class', type=int, default=6)
    return parser


def do_training_validation(argv):
    opts, args = getopt.getopt(argv, "hd:", ["data_root="])
    try:
        opts, args = getopt.getopt(argv, "hd:", ["data_root="])
    except getopt.GetoptError:
        print ('train.py -d <data_root>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print ('train.py -d <data_root>')
            sys.exit()
        elif opt in ("-d", "--data_root"):
            data_root = arg
    print ('data_root is "', data_root)

    pl.seed_everything(21)    
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = configuration_parser(parser)
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    # print(args)
    # init model
    hidden_dim = 50
    model = ActionClassificationLSTM(WINDOW_SIZE, hidden_dim, learning_rate=args.learning_rate)
    data_module = PoseDataModule(data_root=data_root, batch_size=args.batch_size)    
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor='val_loss')
    lr_monitor = LearningRateMonitor(logging_interval='step')    
    # most basic trainer, uses good defaults
    trainer = pl.Trainer.from_argparse_args(args,
        # fast_dev_run=True,
        max_epochs=args.epochs, 
        deterministic=True, 
        gpus=1, 
        progress_bar_refresh_rate=1, 
        callbacks=[EarlyStopping(monitor='train_loss', patience=15), checkpoint_callback, lr_monitor])    
    trainer.fit(model, data_module)    
    return model

if __name__ == '__main__':
    do_training_validation(sys.argv[1:])
