#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Haddy
@file: qlearning.py
@time: 2020-06-17 09:56
@desc:
'''
import gym
import numpy as np
import time

class QLearningAgent(object):
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        self.act_n = act_n  # 动作维度，有几个动作可选
        self.lr = learning_rate  # 学习率
        self.gamma = gamma  # reward的衰减率
        self.epsilon = e_greed  # 按一定概率随机选动作
        self.Q = np.ones((obs_n, act_n))

    # 根据输入观察值，采样输出的动作值，带探索
    def sample(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)  # 一定机率随机探索
        return action

    # 根据输入观察值，预测输出的动作值
    def predict(self, obs):
        #Q-learning code
        Q_list = self.Q[obs, :]  # 取周围4个
        maxQ = np.max(Q_list)
        action_lsit = np.where(Q_list == maxQ)[0]
        action = np.random.choice(action_lsit)
        return action

    # 学习方法，也就是更新Q-table的方法
    def learn(self, obs, action, reward, next_obs, next_action, done):
        """ on-policy
            obs: 交互前的obs, s_t
            action: 本次交互选择的action, a_t
            reward: 本次动作获得的奖励r
            next_obs: 本次交互后的obs, s_t+1
            next_action: 根据当前Q表格, 针对next_obs会选择的动作, a_t+1
            done: episode是否结束
        """
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * np.max(self.Q[next_obs, :])  #q-learning

        self.Q[obs, action] += self.lr * (target_Q - predict_Q)

    # 保存Q表格数据到文件
    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    # 从文件中读取数据到Q表格中
    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')