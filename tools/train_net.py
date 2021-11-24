# -*- encoding: utf-8 -*-
'''
@File    :   train_net.py
@Time    :   2021/11/24 13:27:14
@Author  :   Evan Chen 
@Contact :   chenh_cnyn@163.com
'''

import os
import random
import torch
import numpy as np
from utils.merge_extra_data import merge_pseudo_data
from utils.sampler import simple_up_sample, simple_down_sample

def train(cfg,
          kf,
          model,
          loss_fn,
          optimizer,
          scheduler,
          train_examples,
          oof_cache_fn,
          *args):
    """
    数据划分与模型oof输出保存
    """
    assert isinstance(train_examples, np.ndarray)
    random.shuffle(train_examples)
    
    if cfg.USE_CUDA:
        model.cuda()
    
    if cfg.TRAIN.USE_CV:
        train_val_idx = kf
    else:
        n = len(train_examples)
        n_train =  int(n * cfg.TRAIN.TRAIN_TEST_SPLIT_RATE)
        train_idx, val_idx = list(range(n_train)), list(range(n_train, n))
        train_val_idx = [(train_idx, val_idx)]
        
    oof = dict()
    for k, (train_idx, val_idx) in enumerate(train_val_idx):
        data_train = train_examples[train_idx]
        data_val = train_examples[val_idx] 
        best_oof_preds = do_train(cfg, model, data_train, data_val, optimizer, scheduler, loss_fn, k)
        for idx in val_idx: 
            oof[idx] = best_oof_preds[idx]
        torch.cuda.empty_cache()

    oof_cache_fn(oof, train_examples)


def do_train(cfg,
             model,
             data_train,
             data_val,
             optimizer,
             scheduler,
             loss_fn,
             KF):
    
    best_epoch = 0
    best_kappa = 0
    best_f1 = 0
    best_score = float('-inf')
    best_oof_preds = None
    
    WEIGHTS_NAME = '{}/{}.model'.format(cfg.CACHE_PATH, KF)
    for epoch in range(cfg.TRAIN.EPOCHS):      
        
        if cfg.TRICK.SIMPLE_UPSAMPLE_RATE:
            data_train = simple_up_sample(data_train, cfg.TRICK.SIMPLE_UPSAMPLE_RATE)
        if cfg.TRICK.SIMPLE_DOWNSAMPLE_RATE:
            data_train = simple_down_sample(data_train, cfg.TRICK.SIMPLE_UPSAMPLE_RATE)
        if cfg.TRICK.USE_PSEUDO_DATA:
            data_train = merge_pseudo_data(data_train)

        do_train_epoch(cfg, model, data_train, optimizer, scheduler, loss_fn)
        
        with torch.no_grad():
            emo_preds, emo_logits, tag_preds, acc, kappa, f1, score = do_val(cfg, model, data_val)
            print('Test-KFold: %d, Epoch: %d, loss: %f, acc: %f, kappa: %f, f1: %f, score: %f' % (KF, epoch, epoch_loss, acc, kappa, f1, score))
            
        if score > best_score:
            best_epoch, best_kappa, best_f1 = epoch, kappa, f1
            best_score = score
            best_oof_preds = (emo_preds, emo_logits, tag_preds)

            if os.path.exists(WEIGHTS_NAME):
                os.remove(WEIGHTS_NAME)

            WEIGHTS_NAME = '{}/{}_{:5.f}.model'.format(cfg.CACHE_PATH, KF, score)
            torch.save(model.state_dict(), WEIGHTS_NAME) 
            
    print('Best-Test-KFold: %d, Epoch: %d, loss: %f, kappa: %f, f1: %f, score: %f' % (KF, best_epoch, 0, best_kappa, best_f1, best_score))
    return best_oof_preds


def do_train_epoch():
    pass

def do_val():
    pass