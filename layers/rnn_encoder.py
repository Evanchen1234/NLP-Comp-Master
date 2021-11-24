# -*- encoding: utf-8 -*-
'''
@File    :   rnn_encoder.py
@Time    :   2021/11/24 18:07:59
@Author  :   Evan Chen 
@Contact :   chenh_cnyn@163.com
'''

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNEncoder(nn.Module):
    def __init__(self, rec_type,
                 bert_out_size,
                 hidden_size,
                 num_layers,
                 dropout,
                 bidirectional):
        super(RNNEncoder, self).__init__()

        if rec_type.lower() == 'lstm':
            self.recurrent_layer = nn.LSTM(input_size=bert_out_size,
                                           hidden_size=hidden_size,
                                           num_layers=num_layers,
                                           dropout=dropout,
                                           bidirectional=bidirectional)
        else:
            self.recurrent_layer = nn.GRU(input_size=bert_out_size,
                                          hidden_size=hidden_size,
                                          num_layers=num_layers,
                                          dropout=dropout,
                                          bidirectional=bidirectional)

    def forward(self, words_emb, seq_len):
        words_emb_packed = pack_padded_sequence(words_emb, seq_len, batch_first=True)
        output, _ = self.recurrent_layer(words_emb_packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output


def make_rnn_encoder(rec_type='lstm',
                           bert_out_size=1024,
                           hidden_size=300,
                           num_layers=2,
                           dropout=0.2,
                           bidirectional=True):

    recurrent_encoder = RNNEncoder(
        rec_type = rec_type,
        bert_out_size=bert_out_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional
    )

    return recurrent_encoder
