# -*- encoding: utf-8 -*-
'''
@File    :   container.py
@Time    :   2021/11/26 17:43:30
@Author  :   Evan Chen 
@Contact :   chenh_cnyn@163.com
'''

class Zerba(object):
    def __init__(self, mark):
        self.mark = mark
        self.pre_c = dict()         
        self.fea_c = dict()         
        self.net_c = dict()         
        self.loss_c = dict()        
        self.optimizer_c = dict()   
        self.scheduler_c = dict()   
        self.metrics_c = dict()
    
    @classmethod
    def __anno(*args, **kw):
        return function(*args, **kw)

    def pre(self, version):
        def _anno(function):
            assert function.__code__.co_argcount == 0, \
                        'Do not set parameter for {} method'.format(function.__name__)
            self.pre_c[version] = function()
            return Zerba.__anno 
        return _anno

    def fea(self, version):
        def _anno(function):
            assert function.__code__.co_argcount == 0, \
                        'Do not set parameter for {} method'.format(function.__name__)
            self.fea_c[version] = function()
            return Zerba.__anno 
        return _anno

    def net(self, version):
            def _anno(function):
                assert function.__code__.co_argcount == 0, \
                            'Do not set parameter for {} method'.format(function.__name__)
                self.net_c[version] = function()
                return Zerba.__anno 
            return _anno

    def loss(self, version):
        def _anno(function):
            assert function.__code__.co_argcount == 0, \
                        'Do not set parameter for {} method'.format(function.__name__)
            self.loss_c[version] = function()
            return Zerba.__anno 
        return _anno

    def optimizer(self, version):
        def _anno(function):
            assert function.__code__.co_argcount == 0, \
                        'Do not set parameter for {} method'.format(function.__name__)
            self.optimizer_c[version] = function()
            return Zerba.__anno 
        return _anno

    def scheduler(self, version):
        def _anno(function):
            assert function.__code__.co_argcount == 0, \
                        'Do not set parameter for {} method'.format(function.__name__)
            self.scheduler_c[version] = function()
            return Zerba.__anno 
        return _anno

    def metrics(self, version):
        def _anno(function):
            assert function.__code__.co_argcount == 0, \
                        'Do not set parameter for {} method'.format(function.__name__)
            self.metrics_c[version] = function()
            return Zerba.__anno 
        return _anno