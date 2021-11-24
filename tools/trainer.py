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
from tqdm import tqdm
from functools import reduce
from layers import PGD, FGM
from utils.dataset import Batch
from utils.merge_extra_data import merge_pseudo_data
from utils.sampler import simple_up_sample, simple_down_sample
from sklearn.metrics import cohen_kappa_score, accuracy_score 
from utils.ner import strict_f1


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
            print('Test-KFold: %d, Epoch: %d,  acc: %f, kappa: %f, f1: %f, score: %f' % (KF, epoch, acc, kappa, f1, score))
            
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

def do_train_epoch(cfg,
                   model,
                   data_train,
                   optimizer,
                   scheduler,
                   loss_fn
):    
    random.shuffle(data_train)
    model.train()
    epoch_loss = .0
    if cfg.TRICK.USE_ADV_TRAINING and cfg.TRICK.ADV_TYPE=='pgd':
        k = 3
        pgd = PGD(model)
        
    for ix in tqdm(range(0, len(data_train), cfg.TRAIN.BATCH_SIZE)):
        if ix + cfg.TRAIN.BATCH_SIZE < len(data_train):
            batch_data = data_train[ix: ix + cfg.TRAIN.BATCH_SIZE]
        else:
            batch_data = data_train[ix:]
        batch_data = sorted(batch_data, key=lambda x: -len(x.input_ids))
        examples = Batch(batch_data)

        optimizer.zero_grad()

        logits, _, tokens_loss = model(examples)
        loss = loss_fn(logits, examples.emo_labels, tokens_loss)
        epoch_loss += loss
        loss.backward()

        if cfg.TRICK.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # default 1

        if cfg.TRICK.USE_ADV_TRAINING and cfg.TRICK.ADV_TYPE=='fgm':
            fgm = FGM(model)
            fgm.attack()
            logits, _, tokens_loss = model(examples)
            loss_adv = loss_fn(logits, examples.emo_labels, tokens_loss)
            loss_adv.backward()
            fgm.restore()
        
        if cfg.TRICK.USE_ADV_TRAINING and cfg.TRICK.ADV_TYPE=='pgd':
            pgd.backup_grad()
            for t in range(k):
                pgd.attack(is_first_attack=(t==0))
                if t != k-1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                logits, _, tokens_loss = model(examples)
                loss_adv = loss_fn(logits, examples.emo_labels, tokens_loss)
                loss_adv.backward() 
            pgd.restore()
    
        optimizer.step()
        scheduler.step()

    return epoch_loss

def do_val(cfg, model, val_examples):
    model.eval()
    
    emo_preds = list()
    tag_preds = list()
    emo_labels = [exam.emo_label for exam in val_examples]
    tag_labels = [exam.tag_label for exam in val_examples]
    emo_logits = list()
    
    for ix in range(0, len(val_examples), cfg.VAL.BATCH_SIZE):
        if ix + cfg.VAL.BATCH_SIZE < len(val_examples):
            batch_data = val_examples[ix: ix + cfg.VAL.BATCH_SIZE]
        else:
            batch_data = val_examples[ix:]
        examples = Batch(batch_data)
        
        logits, tokens_out, _ = model(examples)
        emo_preds += list(logits.max(1)[1].cpu().numpy())
        tag_preds += tokens_out
        emo_logits.append(logits.cpu().numpy())
        
    emo_logits = reduce(lambda x,y : np.r_[x, y], emo_logits)
    
    acc = accuracy_score(emo_labels, emo_preds)
    kappa = cohen_kappa_score(emo_labels, emo_preds)
    f1 = strict_f1(tag_preds, tag_labels)
    kappa=0
    #f1=0
    score = 0.5*kappa + 0.5*f1
    return emo_preds, emo_logits, tag_preds, acc, kappa, f1, score