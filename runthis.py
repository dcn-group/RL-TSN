#from dqn.RL_brain import DQN
from dqn.RL_brain import DeepQNetwork
from RL_env.envs import ScheduleEnv
import matplotlib.pyplot as plt
import numpy as np

def run_maze():
    print('\nCollecting experience...')
    step = 0
    reward_record:list = []
    for episode in range(700):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)
            print('Action:',action)

            # RL take action and get next observation and reward

            observation_, reward, done = env.step(action)
            if  (step > 0) and (step % 22 == 0):
                reward_record.append(reward)
            print('Reward:',reward)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            #step += 1
            if done:
                break
            step += 1
        print("eposide:",episode)
    plt.plot(np.arange(len(reward_record)/2), reward_record[::2])
    plt.ylabel('Reward')
    plt.xlabel('training eposide')
    plt.show()

    # end of game
    print('game over')
    #env.destroy()


if __name__ == "__main__":
    # maze game
    env = ScheduleEnv()
    b = env.n_features
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )
    run_maze()
    #env.after(100, run_maze)
    #env.mainloop()
    RL.plot_cost()

