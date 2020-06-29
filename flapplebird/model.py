#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Haddy
@file: model.py
@time: 2020-06-17 13:44
@desc:
'''

import parl
from parl import layers  # 封装了 paddle.fluid.layers 的API

#需要自己写

class Model(parl.Model):
    def __init__(self, act_dim):
        hid1_size = 128
        hid2_size = 128
        # 3层全连接网络
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act=None)

    def value(self, obs):
        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        Q = self.fc3(h2)
        return Q
