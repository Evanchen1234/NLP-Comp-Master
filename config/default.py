# -*- encoding: utf-8 -*-
'''
@File    :   default.py
@Time    :   2021/11/24 10:25:30
@Author  :   Evan Chen 
@Contact :   chenh_cnyn@163.com
'''

from yacs.config import CfgNode as CN


_C = CN()

# General
_C.SEED = 2021      # 全局随机种子
_C.USE_CUDA = False # 是否采用GPU训练
_C.DO_TRAIN = True  # 是否训练模型
_C.DO_VAL = True    # 是否验证模型
_C.DO_TEST = True   # 是否输出测试集结果
_C.MAX_LEN = 512    # 输入文本统一最大长度
_C.TRAIN_PATH = 'raw_data/train/.'    # 训练集位置
_C.TEST_PATH = 'raw_data/test/.'    # 训练集位置

# Classifier
_C.CLS = CN()
_C.CLS.NUM_LABEL = 1    # 单输出的二分类

# Ner
_C.NER = CN()
_C.NER.NUM_TAG = 17     # 标签数量

# Model
_C.MODEL = CN()
_C.MODEL.MODEL_NAME = 'classifier'                    # 模型名称 
_C.MODEL.ENCODER_TYPE = 'bert'                        # encoder类型，bert，nezha，zen...
_C.MODEL.PRETRAIN_PATH = 'pt_models/bert_base_wwm'    # encoder的预先训练模型文件夹地址

# Train
_C.TRAIN = CN()
_C.TRAIN.USE_CV = True      # 是否使用交叉验证
_C.TRAIN.K_SIZE = 5         # 5折
_C.TRAIN.EPOCHS = 10        # 迭代轮次
_C.TRAIN.LR = 2e-5          # 基础学习率
_C.TRAIN.BATCH_SIZE = 32    # 单次更新数量 
_C.TRAIN.EARLY_STOP_ROUNDS = 20     # 验证集合20次不提升则停止学习
_C.TRAIN.TRAIN_PRINT_STEP = 20      # 显示训练信息

#Trick
_C.TRICK = CN()
_C.TRICK.USE_ADV_TRAINING = False   # 是否加入对抗训练
_C.TRICK.ADV_TYPE = 'pgd'           # 对抗训练类型， fgm，pgd
_C.TRICK.USE_GRAD_CLIP = True       # 是否进行梯度裁剪
_C.TRICK.USE_DIFF_LR = True         # 是否使用差分学习率
_C.TRICK.BERT_LR = 3e-6             # 设置encoder的学习率

# Validate
_C.VAL = CN()
_C.VAL.BATCH_SIZE = 32      # 单次更新数量
_C.VAL.VAL_STEP = 200       # 经过VAL_STEP训练后进行一次验证
_C.VAL.VAL_PRINT_STEP = 200 # 显示验证信息

# Test
_C.TS = CN()
_C.TS.BATCH_SIZE = 32       # 单次推断数量
_C.TS.RET_ROOT = 'submit'   # 结果文件夹