# -*- encoding: utf-8 -*-
'''
@File    :   bert_encoder.py
@Time    :   2021/11/24 17:49:40
@Author  :   Evan Chen 
@Contact :   chenh_cnyn@163.com
'''

from transformers import BertModel, ElectraModel


def make_bert_encoder(bert_name='nezha', pt_path='pretrained_models/nezha_large_pytorch'):
    bert_encoder = None
    if bert_name == 'bert':
        bert_encoder = BertModel.from_pretrained(pt_path)
    
    if bert_name == 'electra':
        bert_encoder = ElectraModel.from_pretrained(pt_path)
    
    assert bert_encoder is not None
    return bert_encoder
