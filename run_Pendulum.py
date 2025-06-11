"""
Dueling DQN & Natural DQN comparison

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""
from dqn.DuelingDQN import DuelingDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from RL_env.envs import ScheduleEnv

env =  ScheduleEnv()
#env = env.unwrapped
#env.seed(1)
MEMORY_SIZE = 3000


sess = tf.Session()
with tf.variable_scope('natural'):
    natural_DQN = DuelingDQN(
        n_actions=env.n_actions, n_features=env.n_features, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, sess=sess, dueling=False)

with tf.variable_scope('dueling'):
    dueling_DQN = DuelingDQN(
        n_actions=env.n_actions, n_features=env.n_features, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, sess=sess, dueling=True, output_graph=True)

sess.run(tf.global_variables_initializer())


def train(RL):
    acc_r = [0]
    total_steps = 0
    reward_record: list = []
    observation = env.reset()
    for episode in range(700):
        # initial observation
        observation = env.reset()
        while True:
            action = RL.choose_action(observation)

            observation_, reward, done = env.step(action)

            #reward /= 10      # normalize to a range of (-1, 0)
            #acc_r.append(reward+acc_r[-1])  # accumulated reward
            if (total_steps > 0) and (total_steps % 22 == 0):
            #    reward_record.append(reward+acc_r[-1])
                 acc_r.append(reward + acc_r[-1])

            RL.store_transition(observation, action, reward, observation_)

        #if total_steps > MEMORY_SIZE:
        #    RL.learn()
            if (total_steps > 200) and (total_steps % 5 == 0):
                RL.learn()

            observation = observation_
            if done:
                break
            total_steps += 1
        print("eposide:", episode)
    return RL.cost_his, acc_r
    plt.plot(np.arange(len(reward_record)/2), reward_record[::2])
    plt.ylabel('Reward')
    plt.xlabel('training eposide')
    plt.show()

#train(natural_DQN)
train(dueling_DQN)
c_dueling, r_dueling = train(dueling_DQN)

plt.figure(1)
#plt.plot(np.array(c_natural), c='r', label='natural')
plt.plot(np.array(c_dueling), c='b', label='dueling')
plt.legend(loc='best')
plt.ylabel('cost')
plt.xlabel('training steps')
plt.grid()


plt.figure(2)
#plt.plot(np.array(r_natural), c='r', label='natural')
plt.plot(np.array(r_dueling), c='b', label='dueling')
plt.legend(loc='best')
plt.ylabel('accumulated reward')
plt.xlabel('training steps')
plt.grid()

plt.show()

