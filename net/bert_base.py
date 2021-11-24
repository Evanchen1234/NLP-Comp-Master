# -*- encoding: utf-8 -*-
'''
@File    :   bert_base.py
@Time    :   2021/11/24 18:00:06
@Author  :   Evan Chen 
@Contact :   chenh_cnyn@163.com
'''

import torch.nn as nn
from layers.bert_encoder import make_bert_encoder


class BertBase(nn.Module):
    def __init__(self, cfg):
        super(BertBase, self).__init__()
        self.bert_encoder = make_bert_encoder(bert_name=cfg.MODEL.ENCODER_TYPE, pt_path=cfg.MODEL.PRETRAIN_PATH)
        self.dropout = nn.Dropout(cfg.MODEL.DROPOUT)
        self.classifier = nn.Linear(cfg.MODEL.BERT_OUT_SIZE, cfg.CLS.NUM_LABEL)

    def forward(self, example):
        _, pooled_output = self.bert_encoder(input_ids=example.input_ids,
                                            attention_mask=example.masks,
                                            token_type_ids=example.token_type_ids)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits.squeeze(dim=-1)
