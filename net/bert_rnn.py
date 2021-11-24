# -*- encoding: utf-8 -*-
'''
@File    :   bert_rnn.py
@Time    :   2021/11/24 18:06:52
@Author  :   Evan Chen 
@Contact :   chenh_cnyn@163.com
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.bert_encoder import make_bert_encoder
from layers.rnn_encoder import make_rnn_encoder


class BertRNN(nn.Module):
    def __init__(self, cfg):
        super(BertRNN, self).__init__()
        self.bert_encoder = make_bert_encoder(bert_name=cfg.MODEL.ENCODER_TYPE, pt_path=cfg.MODEL.PRETRAIN_PATH)

        self.rnn_encoder = make_rnn_encoder(cfg.MODEL.RNN_REC_TYPE,
                                            cfg.MODEL.BERT_OUT_SIZE,
                                            cfg.MODEL.RNN_HIDDEN_SIZE,
                                            cfg.MODEL.RNN_HIDDEN_LAYERS,
                                            cfg.MODEL.DROPOUT,
                                            cfg.MODEL.RNN_BIDI)
        self.dropout = nn.Dropout(cfg.MODEL.DROPOUT)
        self.tanh = nn.Tanh()

        # Linear layer to get "convolution output" to be passed to Pooling Layer
        self.W = nn.Linear(cfg.MODEL.BERT_OUT_SIZE + 2 * cfg.MODEL.RNN_HIDDEN_SIZE, cfg.MODEL.RNN_HIDDEN_SIZE)
        self.fc = nn.Linear(cfg.MODEL.RNN_HIDDEN_SIZE, cfg.CLS.NUM_LABEL)

    def forward(self, example):
        word_out = self.bert_encoder(input_ids=example.input_ids, attention_mask=example.masks,
                                    token_type_ids=example.token_type_ids)[0]  # tmp2
        # word_out = self.bert_encoder(input_ids=example.input_ids, attention_mask=example.masks)[0] # tmp

        lstm_out = self.rnn_encoder(word_out, example.input_len)
        # lstm_out.shape = (seq_len, batch_size, 2 * hidden_size)

        input_features = torch.cat([lstm_out, word_out], 2)
        # final_features.shape = (batch_size, seq_len, embed_size + 2*hidden_size)
        
        linear_output = self.tanh(self.W(input_features))
        # linear_output.shape = (batch_size, seq_len, hidden_size_linear)

        linear_output = linear_output.permute(0, 2, 1)  # Reshaping fot max_pool
        max_out_features = F.max_pool1d(linear_output, linear_output.shape[2]).squeeze(2)
        # max_out_features.shape = (batch_size, hidden_size_linear)

        max_out_features = self.dropout(max_out_features)
        final_out = self.fc(max_out_features)
        return final_out.squeeze(dim=-1), [[]],  0
