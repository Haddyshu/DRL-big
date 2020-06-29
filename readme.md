# 训练游戏
FlapplyBird

Paddle
# 训练环境
[1]使用[PLE](https://github.com/ntasfi/PyGame-Learning-Environment)作为训练的环境,只需要从链接clone下该仓库然后安装就可以使用，里面集成了gym-based的强化学习环境。

[2]使用[Orbit](https://github.com/shivaverma/Orbit)作为训练的环境,只需要从链接clone下该仓库，然后将Paddle文件夹下面的paddle.py复制想项目中即可使用。
# 训练框架
使用百度的[PARL](https://github.com/PaddlePaddle/PARL)框架作为代码的框架。该框架主要实现三个部分：

	Model
	Model is abstracted to construct the forward network which defines a policy network or critic network given state as input.
	
	Algorithm
	Algorithm describes the mechanism to update parameters in Model and often contains at least one model.
	
	Agent
	Agent, a data bridge between the environment and the algorithm, is responsible for data I/O with the outside environment and describes data preprocessing before feeding data into the training process.

使用起来简单方便，代码和环境解耦，可以实现即插即用。本次代码使用的DQN代码即使在强化学习七日打卡营上面使用的DQN代码，稍作修改即可在本次实验上面使用。

# Result
最终小鸟能够正常的躲避障碍，能够达到2k+的分数

## [FlappleBird]
![result-bird](./files/bird.gif)

## [Paddle]
![result-bird](./files/paddle.gif)
# how to run
安装好ple后,使用如下命令安装项目依赖

	pip install -r requirements.txt
	then:
	python flapplebird/main.py
	python Paddle/train.py


