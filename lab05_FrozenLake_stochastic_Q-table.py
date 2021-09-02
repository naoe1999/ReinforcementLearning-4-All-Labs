import numpy as np
import matplotlib.pyplot as plt
import gym
# from gym.envs.registration import register


def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return np.random.choice(indices)


def e_greedy(e, action_space, q_vector):
    if np.random.rand(1) < e:
        action = action_space.sample()
    else:
        action = rargmax(q_vector)
    return action


def noisy_action(q_vector, decay_rate):
    noise_vector = np.random.randn(*q_vector.shape) * decay_rate
    action = np.argmax(q_vector + noise_vector)
    return action


if __name__ == '__main__':

    env = gym.make('FrozenLake-v0')     # is_slippery: True
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])
    learning_rate = 0.85
    dis = 0.99
    num_episodes = 2000

    rList = []
    for i in range(num_episodes):
        state = env.reset()
        rAll = 0
        done = False
        e = .1 / ((i//100)+1)

        while not done:
            # select an action:  e-greedy or noise
            # action = e_greedy(e, env.action_space, Q[state, :])
            action = noisy_action(Q[state, :], e)

            # execute and receive immediate reward
            new_state, reward, done, _ = env.step(action)
            rAll += reward

            # update Q-table
            # Q[state, action] = reward + dis * np.max(Q[new_state, :])
            Q[state, action] = (1-learning_rate) * Q[state, action] \
                + learning_rate * (reward + dis * np.max(Q[new_state, :]))

            # update state
            state = new_state

        rList.append(rAll)

    print('Success rate: ' + str(sum(rList) / num_episodes))
    print('Final Q-Table Values')
    print('  LEFT       DOWN       RIGHT      UP')
    print(Q)

    plt.bar(range(len(rList)), rList)
    plt.show()

