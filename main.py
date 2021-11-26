# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2021/11/24 18:24:12
@Author  :   Evan Chen 
@Contact :   chenh_cnyn@163.com
'''

import os
import sys
sys.path.append(os.getcwd())
from config import cfg
from zerba import Zerba

app = Zerba('nlp竞赛模版')
