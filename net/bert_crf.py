# -*- encoding: utf-8 -*-
'''
@File    :   bert_crf.py
@Time    :   2021/11/24 17:52:05
@Author  :   Evan Chen 
@Contact :   chenh_cnyn@163.com
'''

import torch.nn as nn
from torchcrf import CRF
from layers.bert_encoder import make_bert_encoder

class BertCRF(nn.Module):
    def __init__(self, cfg):
        super(BertCRF, self).__init__()
        self.bert_encoder = make_bert_encoder(bert_name=cfg.MODEL.ENCODER_TYPE, pt_path=cfg.MODEL.PRETRAIN_PATH)
        
        # meo
        self.emo_classifier = nn.Sequential(
            nn.Dropout(cfg.MODEL.DROPOUT),
            nn.Linear(cfg.MODEL.BERT_OUT_SIZE, cfg.CLS.NUM_LABEL)
        )

        # crd model
        self.mid_linear = nn.Sequential(
            nn.Linear(cfg.MODEL.BERT_OUT_SIZE, cfg.MODEL.MID_LINEAR_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.MODEL.DROPOUT)
        )
        self.tag_classifier = nn.Linear(cfg.MODEL.MID_LINEAR_DIM, cfg.NER.NUM_TAG)
        self.crf_module = CRF(cfg.NER.NUM_TAG, batch_first=True)
    
    def forward(self, example):
        bert_outputs = self.bert_encoder(input_ids=example.input_ids, attention_mask=example.masks,
                                    token_type_ids=example.token_type_ids)
        logits = self.emo_classifier(bert_outputs[1])
        
        seq_out = bert_outputs[0]
        seq_out = self.mid_linear(seq_out)
        emissions = self.tag_classifier(seq_out)

        tokens_out = self.crf_module.decode(emissions=emissions, mask=example.masks.byte())
        tokens_loss = None
        if example.train:
            tokens_loss = -1 * self.crf_module(emissions=emissions, tags=example.tag_labels.long(), mask=example.masks.byte(), reduction='mean')

        return (logits, tokens_out, tokens_loss)
