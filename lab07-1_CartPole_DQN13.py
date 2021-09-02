import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from utils.action import e_greedy, noisy_action


class DQN:
    def __init__(self, input_size, output_size, l_rate=1e-1, name='main'):
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        self._build_network(l_rate=l_rate)

    def _build_network(self, h_size=10, l_rate=1e-1):
        self.W1 = tf.Variable(tf.random.normal([self.input_size, h_size], 0, 0.01, dtype=tf.float64))
        self.W2 = tf.Variable(tf.random.normal([h_size, self.output_size], 0, 0.01, dtype=tf.float64))
        self.variables = [self.W1, self.W2]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.forward(x).numpy()

    @tf.function
    def forward(self, x):
        h = tf.tanh(tf.matmul(x, self.W1))
        return tf.matmul(h, self.W2)

    @tf.function
    def update(self, x, y):
        with tf.GradientTape() as tape:
            Qpred = self.forward(x)
            loss = tf.reduce_sum(tf.square(y - Qpred))
        gradients = tape.gradient(loss, self.variables)
        self.optimizer.apply_gradients(zip(gradients, self.variables))
        return loss


def simple_replay_train(dqn, train_batch, discount=0.9):
    x_stack = np.empty(0).reshape(0, dqn.input_size)
    y_stack = np.empty(0).reshape(0, dqn.output_size)

    for state, action, reward, next_state, done in train_batch:
        Q = dqn.predict(state)

        if done:
            Q[0, action] = reward
        else:
            Qs1 = dqn.predict(next_state)
            Q[0, action] = reward + discount * np.max(Qs1)

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    return dqn.update(x_stack, y_stack)


def bot_play(env, dqn):
    s = env.reset()
    reward_sum = 0
    done = False
    while not done:
        env.render()
        a = np.argmax(dqn.predict(s))
        s, reward, done, _ = env.step(a)
        reward_sum += reward
    print('Total score:', reward_sum)


def main():
    env = gym.make('CartPole-v0')
    env._max_episode_steps = 20000  # 늘려주지 않으면 200에서 끝남. 제대로 학습 안됨!!

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    BATCH_SIZE = 10
    LEARNING_RATE = 0.1
    DISCOUNT_RATIO = 0.9
    REPLAY_MEMORY = 50000
    MAX_EPISODES = 2000

    replay_buffer = deque()
    reward_list = []

    mainDQN = DQN(input_size, output_size, l_rate=LEARNING_RATE)

    for episode in range(MAX_EPISODES):
        state = env.reset()

        done = False
        e = 1. / ((episode / 10) + 1)
        step_count = 0

        while not done:
            # select an action
            Qs = mainDQN.predict(state)
            action = e_greedy(e, env.action_space, Qs)

            # get new state and reward
            next_state, reward, done, _ = env.step(action)
            if done:
                reward = -100

            # save the experience to our buffer
            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > REPLAY_MEMORY:
                replay_buffer.popleft()

            # update state
            state = next_state

            step_count += 1
            if step_count > 10000:
                break

        print('Episode: {}, steps: {}'.format(episode, step_count))
        reward_list.append(step_count)

        # train DQN (50 steps for every 10 episodes)
        if episode % 10 == 1:
            print('Training DQN ... ', end='', flush=True)
            llist = []
            for i in range(50):
                minibatch = random.sample(replay_buffer, BATCH_SIZE)
                loss = simple_replay_train(mainDQN, minibatch, DISCOUNT_RATIO)
                llist.append(loss.numpy())
            print('done. Average loss: ', sum(llist)/len(llist))

    # training history
    print('Maximum reward sum:', max(reward_list))
    print('Average reward sum:', sum(reward_list) / len(reward_list))

    # test
    bot_play(env, mainDQN)
    env.close()


if __name__ == '__main__':
    main()

