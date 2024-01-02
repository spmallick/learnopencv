"""
'''
///////////////////////////////////////
3D LiDAR Object Detection - ADAS
Pranav Durai
//////////////////////////////////////
'''
# Description: This script for logging

"""
import os
import logging

class Logger():
    """
        Create logger to save logs during training
        Args:
            logs_dir:
            saved_fn:

        Returns:

        """

    def __init__(self, logs_dir, saved_fn):
        logger_fn = 'logger_{}.txt'.format(saved_fn)
        logger_path = os.path.join(logs_dir, logger_fn)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # formatter = logging.Formatter('%(asctime)s:File %(module)s.py:Func %(funcName)s:Line %(lineno)d:%(levelname)s: %(message)s')
        formatter = logging.Formatter(
            '%(asctime)s: %(module)s.py - %(funcName)s(), at Line %(lineno)d:%(levelname)s:\n%(message)s')

        file_handler = logging.FileHandler(logger_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def info(self, message):
        self.logger.info(message)
