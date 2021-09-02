import numpy as np
import random as pr
import matplotlib.pyplot as plt
import gym
from gym.envs.registration import register


def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)


if __name__ == '__main__':
    register(
        id='FrozenLake-v3',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False}
    )

    env = gym.make("FrozenLake-v3")
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])
    num_episodes = 2000

    rList = []
    for i in range(num_episodes):
        state = env.reset()
        rAll = 0
        done = False

        while not done:
            # select an action
            action = rargmax(Q[state, :])

            # execute and receive immediate reward
            new_state, reward, done, _ = env.step(action)
            rAll += reward

            # update Q-table
            Q[state, action] = reward + np.max(Q[new_state, :])

            # s_t <-- s_t+1
            state = new_state

        rList.append(rAll)

    print('Success rate: ' + str(sum(rList) / num_episodes))
    print('Final Q-Table Values')
    print('  L  D  R  U')
    print(Q)

    plt.bar(range(len(rList)), rList)
    plt.show()

