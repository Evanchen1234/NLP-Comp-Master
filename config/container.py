# -*- encoding: utf-8 -*-
'''
@File    :   container.py
@Time    :   2021/11/26 17:43:30
@Author  :   Evan Chen 
@Contact :   chenh_cnyn@163.com
'''

class Container(object):    
    def add(self, name, instance):
        self.__dict__[name] = instance