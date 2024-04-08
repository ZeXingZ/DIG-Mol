import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import random

class AverageMeter(object):
    """Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):#需要将所有变量清零的时候，调用
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):#需要更新某个变量的时候,调用
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count