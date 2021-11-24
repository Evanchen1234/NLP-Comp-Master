# -*- encoding: utf-8 -*-
'''
@File    :   load_tokenizer.py
@Time    :   2021/11/24 12:02:32
@Author  :   Evan Chen 
@Contact :   chenh_cnyn@163.com
'''

from transformers import BertTokenizer, ElectraTokenizer


def load_tokenizer(args):
    tokenizer = None
    if args.bert_name.lower() == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    elif args.bert_name.lower() == 'electra':
        tokenizer = ElectraTokenizer.from_pretrained(args.bert_path)
    assert tokenizer is not None, 'load tokenizer error'
    return tokenizer