import gym
import random
import numpy as np
from collections import deque
from utils.action import e_greedy
from models.dqn import DQN


def replay_train(dqn_main, dqn_target, train_batch, discount=0.9):
    x_stack = []
    y_stack = []

    for state, action, reward, next_state, done in train_batch:
        Q = dqn_main.predict(state)

        if done:
            Q[0, action] = reward
        else:
            Qs1 = dqn_target.predict(next_state)    # this comes from target DQN
            Q[0, action] = reward + discount * np.max(Qs1)

        x_stack.append(state)
        y_stack.append(Q)

    xs = np.array(x_stack)
    ys = np.array(y_stack)
    loss = dqn_main.update(xs, ys)
    return loss


def bot_play(dqn):
    s = env.reset()
    reward_sum = 0
    done = False
    while not done:
        env.render()
        a = np.argmax(dqn.predict(s))
        s, reward, done, _ = env.step(a)
        reward_sum += reward
    print('Total score:', reward_sum)


def train(dqn_main, dqn_target):
    max_episodes = MAX_EPISODES
    replay_buffer = deque()

    for episode in range(max_episodes):
        state = env.reset()

        done = False
        e = 1. / ((episode / 10) + 1)
        step_count = 0
        total_reward = 0

        while not done:
            # select an action
            Qs = dqn_main.predict(state)
            action = e_greedy(e, env.action_space, Qs)

            # get new state and reward
            next_state, reward, done, _ = env.step(action)
            # if done:
            #     reward = -100

            # save the experience to our buffer
            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > REPLAY_MEMORY:
                replay_buffer.popleft()

            # update state
            state = next_state

            total_reward += reward
            step_count += 1
            if step_count > 10000:
                break

        print('Episode: {}, steps: {}, score: {}'.format(episode, step_count, total_reward))

        # train DQN (50 steps for every 10 episodes)
        if episode % 10 == 0:
            print('Training DQN ... ', end='', flush=True)
            llist = []
            for i in range(50):
                minibatch = random.sample(replay_buffer, BATCH_SIZE)
                loss = replay_train(dqn_main, dqn_target, minibatch, DISCOUNT_RATIO)
                llist.append(loss)
            print('done. Average loss: ', sum(llist)/len(llist))

            # copy network
            mainDQN.copy_to(targetDQN)

    # end of training


if __name__ == '__main__':

    env = gym.make('BreakoutNoFrameskip-v4')

    print('max steps:', env._max_episode_steps)
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    input_shape = env.observation_space.shape
    output_size = env.action_space.n

    BATCH_SIZE = 32
    LEARNING_RATE = 0.1
    DISCOUNT_RATIO = 0.9
    REPLAY_MEMORY = 50000
    MAX_EPISODES = 2000

    mainDQN = DQN(input_shape, output_size, l_rate=LEARNING_RATE, name='main')
    targetDQN = DQN(input_shape, output_size, name='target')
    mainDQN.copy_to(targetDQN)

    train(mainDQN, targetDQN)

    bot_play(mainDQN)
    env.close()

