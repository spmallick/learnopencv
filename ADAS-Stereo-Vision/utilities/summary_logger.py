#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import logging
import os

from tensorboardX import SummaryWriter


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory
        self.writer = SummaryWriter(log_dir=os.path.join(self.directory))

    def config_logger(self, epoch):
        # create logger with 'spam_application'
        logger = logging.getLogger(str(epoch))
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(os.path.join(self.directory, 'epoch_' + str(epoch) + '.log'))
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger
