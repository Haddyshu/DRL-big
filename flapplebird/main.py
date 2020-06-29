#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Haddy
@file: main.py
@time: 2020-06-29 23:30
@desc:
'''
from ple.games.flappybird import FlappyBird
from ple import PLE

import os
import numpy as np

import paddle.fluid as fluid
import parl
from parl import layers
from parl.utils import logger
from parl.algorithms import PolicyGradient
# 策略梯度方法求解RL
LEARNING_RATE = 1e-3

class Model(parl.Model):
    def __init__(self, act_dim):
        act_dim = act_dim
        hid1_size = act_dim * 10
        # self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc1 = layers.fc(size=hid1_size, act='tanh')
        self.fc2 = layers.fc(size=act_dim, act='softmax')

    def forward(self, obs):  # 可直接用 model = Model(5); model(obs)调用
        out = self.fc1(obs)
        out = self.fc2(out)
        return out

class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.act_prob = self.alg.predict(obs)

        with fluid.program_guard(
                self.learn_program):  # 搭建计算图用于 更新policy网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(name='act', shape=[1], dtype='int64')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            self.cost = self.alg.learn(obs, act, reward)

    def sample(self, obs):
        obs = np.expand_dims(obs, axis=0)  # 增加一维维度
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)  # 减少一维维度
        act = np.random.choice(range(self.act_dim), p=act_prob)  # 根据动作概率选取动作
        return act

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)
        act = np.argmax(act_prob)  # 根据动作概率选择概率最高的动作
        return act

    def learn(self, obs, act, reward):
        act = np.expand_dims(act, axis=-1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int64'),
            'reward': reward.astype('float32')
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]
        return cost

# 根据一个episode的每个step的reward列表，计算每一个Step的Gt
def calc_reward_to_go(reward_list, gamma=1.0):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        reward_list[i] += gamma * reward_list[i + 1]  # Gt
    return np.array(reward_list)

def run_episode(ple_env, agent):
    obs_list, action_list, reward_list = [], [], []
    ple_env.reset_game()
    obs = list(ple_env.getGameState().values())
    while True:
        obs_list.append(obs)
        action_index = agent.sample(obs) # 采样动作
        action_list.append(action_index)
        action = ple_env.getActionSet()[action_index]
        reward = ple_env.act(action)
        reward_list.append(reward)

        if ple_env.game_over():
            break
    return obs_list, action_list, reward_list

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(ple_env, agent, render=False):
    eval_reward = []
    for i in range(5):
        ple_env.reset_game()
        obs = list(ple_env.getGameState().values())
        episode_reward = 0
        while True:
            action_index = agent.predict(obs) # 选取最优动作
            action = ple_env.getActionSet()[action_index]
            reward = ple_env.act(action)
            episode_reward += reward
            if render:
                ple_env.getScreenRGB()
            if ple_env.game_over():
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

# 创建环境
game = FlappyBird()
p = PLE(game, fps=30, display_screen=True,force_fps=True)
p.init()
# 根据parl框架构建agent
print(p.getActionSet())
act_dim = len(p.getActionSet())
states = len(p.getGameState())
model = Model(act_dim=act_dim)
alg = PolicyGradient(model, lr=LEARNING_RATE)
agent = Agent(alg, obs_dim=states, act_dim=act_dim)

# 加载模型
if os.path.exists('./model.ckpt'):
    agent.restore('./model.ckpt')
    # run_episode(env, agent, train_or_test='test', render=True)
    # exit()

for i in range(10000):
    obs_list, action_list, reward_list = run_episode(p, agent)
    if i % 10 == 0:
        logger.info("Episode {}, Reward Sum {}.".format(
            i, sum(reward_list)))

    batch_obs = np.array(obs_list)
    batch_action = np.array(action_list)
    batch_reward = calc_reward_to_go(reward_list)

    agent.learn(batch_obs, batch_action, batch_reward)
    if (i + 1) % 100 == 0:
        total_reward = evaluate(p, agent, render=False) # render=True 查看渲染效果，需要在本地运行，AIStudio无法显示
        logger.info('Test reward: {}'.format(total_reward))

# 保存模型到文件 ./model.ckpt
agent.save('./model.ckpt')