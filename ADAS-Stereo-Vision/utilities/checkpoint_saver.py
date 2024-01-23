#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import glob
import os

import torch
from pygit2 import Repository


class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.checkpoint)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        self.save_experiment_config()

    def save_checkpoint(self, state, filename='model.pth.tar', write_best=True):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

        best_pred = state['best_pred']
        if write_best:
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))

    def save_experiment_config(self):
        with open(os.path.join(self.experiment_dir, 'parameters.txt'), 'w') as file:
            config_dict = vars(self.args)
            for k in vars(self.args):
                file.write(f"{k}={config_dict[k]} \n")
            repo = Repository('.')
            file.write(f"git-repo={repo.head.shorthand} \n")
