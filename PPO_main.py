#import gym
#import numpy as np
#import torch

from algorithms.PPO import ActorCriticAgent
from utils import plotLearning
import os
import yaml
import csv
import time
import argparse
from types import SimpleNamespace as SN
from RL_env.envs import ScheduleEnv
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

with open(os.path.join(os.path.dirname(__file__), "config", "PPO.yaml"), "r") as f:
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
    #test_env  = ScheduleEnv()

    #train_env.seed(args.seed)
    #test_env.seed(args.seed)
    #np.random.seed(args.seed)
    #torch.manual_seed(args.seed)

    args.state_dim = train_env.n_features
    args.action_dim = train_env.n_actions

    #agent = ActorCriticAgent(args)
    #episode_reward_history = []
    #current_step_history = []
    #current_steps = 0

    agent = ActorCriticAgent(args)
    episode_reward_history = []
    current_step_history = []
    current_steps = 0
    pre_reward = 0

    a = 0.85
    # Build a tensorboard
    writer = SummaryWriter(log_dir='runs/1')
    start_time: time.process_time = time.perf_counter()  # start time
    while current_steps < args.total_steps:
        state = train_env.reset()
        done = False
        while not done:
            current_steps += 1
            action = agent.select_action(state)
            next_state, reward, done = train_env.step(action)
            # buffer
            agent.buffer.states.append(state)
            agent.buffer.actions.append(action)
            agent.buffer.rewards.append(reward)
            agent.buffer.next_states.append(next_state)
            agent.buffer.masks.append(1 - int(done))

            state = next_state

            if current_steps % args.train_steps == 0:
                agent.train()
            if current_steps % args.test_steps == 0:
                average_episode_reward = tagent(train_env, agent)
                average_episode_reward = -average_episode_reward
                # 第一种reward平滑方式
                if current_steps == 1000:
                    pre_reward = average_episode_reward
                else:
                    pre_reward = episode_reward_history[-1]
                average_episode_reward = average_episode_reward * a + (1-a) * pre_reward
                # 第二种reward平滑方式

                episode_reward_history.append(average_episode_reward)
                current_step_history.append(current_steps)
                print(f'| step : {current_steps:6} | Episode Reward: {average_episode_reward:5.5f} |')
                writer.add_scalar('step_rewards', episode_reward_history[-1], global_step=current_steps)
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
    # writer.add_scalar('step_rewards', episode_reward_history, global_step=args.total_steps)
    # writer.close()

