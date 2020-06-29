#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Haddy
@file: main_pd.py
@time: 2020-06-17 09:26
@desc:
'''
from big.jump.qlearning import QLearningAgent
from big.jump.jump import JumpGame

def run_episode(env, agent, render=False):
    total_steps = 0 # 记录每个episode走了多少step
    total_reward = 0

    obs = env.reset() # 重置环境, 重新开一局（即开始新的一个episode）
    print('action in obs', obs)
    action = agent.sample(obs) # 根据算法选择一个动作
    print('action in run',action)

    while True:
        reward, next_obs ,done= env.step(action) # 与环境进行一个交互
        next_action = agent.sample(4.5) # 根据算法选择一个动作 q-learning 不用这一步
        # 训练 Sarsa 算法
        agent.learn(obs, action, reward, next_obs, next_action, done)

        action = next_action
        obs = next_obs  # 存储上一个观察值
        total_reward += reward
        total_steps += 1 # 计算step数
        if done:
            break
    return total_reward, total_steps

def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs) # greedy
        reward, next_obs, done = env.step(action)  # 与环境进行一个交互
        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


def qlearning_run():
    # 创建一个agent实例，输入超参数
    agent = QLearningAgent(
        obs_n=2,
        act_n=len(obs),
        learning_rate=0.3,
        gamma=0.9,
        e_greed=0.09)
    return agent


if __name__ == '__main__':
    env = JumpGame()
    obs = env.reset()

    agent = qlearning_run()
    # 训练500个episode，打印每个episode的分数
    is_render = False
    for episode in range(500):
        ep_reward, ep_steps = run_episode(env, agent, is_render)
        print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))
        if episode%20==0:
            is_render = True
        else:
            is_render = False

    # 全部训练结束，查看算法效果
    test_reward = test_episode(env, agent)
    print('test reward = %.1f' % (test_reward))