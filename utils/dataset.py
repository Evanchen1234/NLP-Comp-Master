# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2021/11/24 15:16:12
@Author  :   Evan Chen 
@Contact :   chenh_cnyn@163.com
'''

import torch


def pad_func(data, pad=0):
    seq_len = [len(i) for i in data]
    max_len = max(seq_len)
    if max_len > 511:
        max_len = 512
    out = []
    for i in data:
        if len(i) < max_len:
            out.append(i + [pad] * (max_len-len(i)))
        else:
            out.append(i[:512])
    return torch.tensor(out).cuda()

class Batch(object):
    def __init__(self, examples, train=True):
        self.train = train
        self.examples = examples
        self.input_ids = pad_func([e.input_ids for e in examples])
        self.token_type_ids = pad_func([e.token_type_id for e in examples])
        self.masks = self.get_mask(self.input_ids)
        self.input_len = torch.tensor([len(e.input_ids) for e in examples]).cuda()

        if train:
            self.tag_labels = pad_func([e.tag_label for e in examples])
            self.emo_labels = torch.tensor([e.emo_label for e in examples]).cuda()
            

    def __len__(self):
        return len(self.examples)

    def get_mask(self, data, pad=0):
        return torch.tensor(data) != pad

    def cal_len(self,x):
        len_seq = 0
        for i in x:
            len_seq + len(i)
        return len_seq