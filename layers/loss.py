# -*- encoding: utf-8 -*-
'''
@File    :   loss.py
@Time    :   2021/11/24 17:47:29
@Author  :   Evan Chen 
@Contact :   chenh_cnyn@163.com
'''

import torch
import torch.nn as nn


class MultiFocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(MultiFocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
 
    def forward(self, input, target):
        N = target.size(0)
        smooth = 1
 
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
 
        intersection = input_flat * target_flat
 
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
        return loss

class MulticlassDiceLoss(nn.Module):
    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()
 
    def forward(self, input, target, weights=None):
 
        C = target.shape[1]
  
        dice = DiceLoss()
        totalLoss = 0
 
        for i in range(C):
            diceLoss = dice(input[:,i], target[:,i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss
 
        return totalLoss