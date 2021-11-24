# -*- encoding: utf-8 -*-
'''
@File    :   ner.py
@Time    :   2021/11/24 12:03:01
@Author  :   Evan Chen 
@Contact :   chenh_cnyn@163.com
'''

from seqeval.metrics import f1_score as seq_f1_score


def bio2bioes_str(tag_str):
    """
    'O O O O O O B-BANK I-BANK O O O O O O B-COMMENTS_ADJ I-COMMENTS_ADJ O B-COMMENTS_N' 
    'O O O O O O B-BANK E-BANK O O O O O O B-COMMENTS_ADJ E-COMMENTS_ADJ O S-COMMENTS_N'
    """
    
    new_tags = []
    for tag in tag_str.split():
        if tag == 'O' or tag.startswith('B-'):
            if new_tags: #对最后一个entity进行修改
                if new_tags[-1].startswith('B-'):
                    new_tags[-1] = 'S-' + new_tags[-1][2:]
                if new_tags[-1].startswith('I-'):
                    new_tags[-1] = 'E-' + new_tags[-1][2:]
        new_tags.append(tag)
            
    if new_tags and new_tags[-1].startswith('B-'):
        new_tags[-1] = 'S-' + new_tags[-1][2:]
    
    if new_tags and new_tags[-1].startswith('I-'):
        new_tags[-1] = 'E-' + new_tags[-1][2:]

    return ' '.join(new_tags)


def bioes2bio_str(tag_str):
    """
    'O O O O O O B-BANK E-BANK O O O O O O B-COMMENTS_ADJ E-COMMENTS_ADJ O S-COMMENTS_N'
    'O O O O O O B-BANK I-BANK O O O O O O B-COMMENTS_ADJ I-COMMENTS_ADJ O B-COMMENTS_N'
    """
    new_tags = []
    for tag in tag_str.split():
        if tag.startswith('E-'):
            tag = 'I-' + tag[2:]
        if tag.startswith('S-'):
            tag = 'B-' + tag[2:]
        new_tags.append(tag)
    return ' '.join(new_tags)

class TagTransmit(object):
    def __init__(self):
        tags = [
            'O',
            'B-BANK',
            'I-BANK',
            'B-PRODUCT',
            'I-PRODUCT',
            'B-COMMENTS_N',
            'I-COMMENTS_N',
            'B-COMMENTS_ADJ',
            'I-COMMENTS_ADJ',
            'E-BANK',
            'E-PRODUCT',
            'E-COMMENTS_N',
            'E-COMMENTS_ADJ',
            'S-BANK',
            'S-PRODUCT',
            'S-COMMENTS_N',
            'S-COMMENTS_ADJ'
            ]
        self.id2tag = {i:j for i, j in enumerate(tags)}
        self.tag2id = {j:i for i, j in enumerate(tags)}
    
    def encode(self, row):
        return [self.tag2id[item] for item in row]

    def decode(self, row):
        return [self.id2tag[item] for item in row]

tt = TagTransmit()
def strict_f1(y_pred, y_true):
    y_pred = [tt.decode(item) for item in y_pred]
    y_true = [tt.decode(item) for item in y_true]
    return seq_f1_score(y_true, y_pred, mode='strict')