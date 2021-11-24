# -*- encoding: utf-8 -*-
'''
@File    :   load_tokenizer.py
@Time    :   2021/11/24 12:02:32
@Author  :   Evan Chen 
@Contact :   chenh_cnyn@163.com
'''

from transformers import BertTokenizer, ElectraTokenizer


def load_tokenizer(cfg):
    tokenizer = None
    if cfg.MODEL.ENCODER_TYPE.lower() == 'bert':
        tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.PRETRAIN_PATH)
    elif cfg.MODEL.ENCODER_TYPE.lower() == 'electra':
        tokenizer = ElectraTokenizer.from_pretrained(cfg.MODEL.PRETRAIN_PATH)
    assert tokenizer is not None, 'load tokenizer error'
    return tokenizer