# -*- encoding: utf-8 -*-
'''
@File    :   classifier.py
@Time    :   2021/11/24 12:06:20
@Author  :   Evan Chen 
@Contact :   chenh_cnyn@163.com
'''

from sklearn.metrics import f1_score, precision_score, recall_score


def bin_search_f1(y_true, y_pred):
    """
    二分类阈值搜索
    """
    best_f1 = 0
    best_t = 0
    p, r = 0, 0
    for i in range(30, 60):
        tres = i / 100
        y_pred_bin = (y_pred > tres).astype(int)
        score = f1_score(y_true, y_pred_bin)
        if score > best_f1:
            best_f1 = score
            best_t = tres
            p = precision_score(y_true, y_pred_bin)
            r = recall_score(y_true, y_pred_bin)
    return p, r, best_f1, best_t