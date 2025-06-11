import gym
import numpy as np
import torch

from algorithms.dueling_double_dqn import DuelingDoubleDQNAgent
from utils import plotLearning
import os
import yaml
import time
import csv
from types import SimpleNamespace as SN
from datetime import datetime
from RL_env.envs import ScheduleEnv

with open(os.path.join(os.path.dirname(__file__), "config", "dueling_double_dqn.yaml"), "r") as f:
    try:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        assert False, "default.yaml error: {}".format(exc)
args = SN(**config_dict)


def tagent(env, agent):
    episode_reward = 0
    for i in range(args.test_num):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_argmax_action(state)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
    return episode_reward / args.test_num


if __name__ == '__main__':
    train_env = ScheduleEnv()
    # train_env = gym.make(args.env_name)
    # test_env = gym.make(args.env_name)
    #
    # train_env.seed(args.seed)
    # test_env.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    #
    # args.state_dim = train_env.observation_space.shape[0]
    # args.action_dim = train_env.action_space.n
    args.state_dim = train_env.n_features
    args.action_dim = train_env.n_actions

    agent = DuelingDoubleDQNAgent(args)
    episode_reward_history = []
    current_step_history = []
    current_steps = 0

    a = 0.8
    start_time: time.process_time = time.perf_counter()  # start time
    while current_steps < args.total_steps:
        state = train_env.reset()
        done = False
        while not done:
            current_steps += 1
            action = agent.select_action(state)
            next_state, reward, done = train_env.step(action)
            # buffer
            agent.buffer.put(state, action, reward, next_state, 1 - int(done))
            state = next_state
            # todo the best train interval
            agent.train()
            if current_steps % args.test_steps == 0:
                average_episode_reward = tagent(train_env, agent)
                average_episode_reward = -average_episode_reward
                # 第一种reward平滑方式
                if current_steps == 1000:
                    pre_reward = average_episode_reward
                else:
                    pre_reward = episode_reward_history[-1]
                average_episode_reward = average_episode_reward * a + (1 - a) * pre_reward

                episode_reward_history.append(average_episode_reward)
                current_step_history.append(current_steps)
                print(f'| step : {current_steps:6} | Episode Reward: {average_episode_reward:5.5f} |')
            if current_steps % args.save_steps == 0:
                agent.save(f'{args.checkpoint_path}/{args.seed}-{args.name}-{current_steps}')

    end_time: time.process_time = time.perf_counter()  # end time
    run_time = end_time - start_time
    print('运行时间:', run_time)
    file_name = f'{args.seed}-{args.name}-{args.env_name}.png'
    plotLearning(episode_reward_history, current_step_history, filename=file_name, window=25)
    with open('data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Step', 'Reward'])
        for i in range(len(current_step_history)):
            writer.writerow([current_step_history[i], round(episode_reward_history[i], 5)])

    # plot_figure(episode_reward_history, "Episode", "Reward", file_name)
