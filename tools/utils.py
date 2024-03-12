import torch
import random
import numpy as np
import os
import logging
import sys

def seed_training_code(manual_seed, strict=False):
    """Control pseudo-randomness for reproducibility.
    :param manual_seed: (int) random-seed
    :param strict: (boolean) if True, cudnn operates in a deterministic manner
    """
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu_to_zero_position(real_gpu_loc):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(real_gpu_loc)

def create_logger(log_dir,std_out=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # Add logging to file handler
    file_handler = logging.FileHandler(os.path.join(log_dir,'log.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Add stdout to also print statements there
    if std_out:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count