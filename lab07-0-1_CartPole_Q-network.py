import gym
import tensorflow as tf
from utils.action import *


class QNet:
    def __init__(self, input_size, output_size, l_rate=1e-1, name='main'):
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        self._build_network(l_rate=l_rate)

    def _build_network(self, l_rate=1e-1):
        self.W = tf.Variable(
            tf.random.normal([self.input_size, self.output_size], 0, 0.1, dtype=tf.float64)
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate)

    @tf.function
    def forward(self, x):
        return tf.matmul(x, self.W)

    @tf.function
    def update(self, x, y):
        with tf.GradientTape() as tape:
            Qpred = self.forward(x)
            loss = tf.reduce_sum(tf.square(y - Qpred))
        gradients = tape.gradient(loss, self.W)
        self.optimizer.apply_gradients([(gradients, self.W)])
        return loss


if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    env._max_episode_steps = 20000

    i_size = env.observation_space.shape[0]
    o_size = env.action_space.n
    lr = 0.1
    qnet = QNet(i_size, o_size, l_rate=lr)

    num_episodes = 2000
    dis = 0.9
    rList = []

    for i in range(num_episodes):
        s = env.reset()

        e = 1. / ((i/10) + 1)
        step_count = 0
        done = False

        while not done:
            step_count += 1

            # preprocess state
            x = s.reshape([1, i_size])

            # get Q vector using Q-Network
            Qs = qnet.forward(x).numpy()

            # select an action
            a = e_greedy(e, env.action_space, Qs)

            # execute the action and receive immediate reward
            s1, reward, done, _ = env.step(a)

            # update Q table
            if done:
                Qs[0, a] = -100
            else:
                x1 = s1.reshape([1, i_size])
                Qs1 = qnet.forward(x1).numpy()
                Qs[0, a] = reward + dis * np.max(Qs1)

            # train Q-Network to fit to the updated Q-function
            qnet.update(x, Qs)

            # update state
            s = s1

        print('Episode: {}, steps: {}'.format(i, step_count))
        rList.append(step_count)

        if len(rList) > 10 and np.mean(rList[-10:]) > 500:
            break

    # training history
    print('Maximum reward sum:', max(rList))
    print('Average reward sum:', sum(rList) / len(rList))

    # test
    observation = env.reset()
    reward_sum = 0
    done = False
    while not done:
        env.render()
        x = observation.reshape([1, i_size])
        Qs = qnet.forward(x).numpy()
        a = np.argmax(Qs)

        observation, reward, done, _ = env.step(a)
        reward_sum += reward

    env.close()
    print('Total score: {}'.format(reward_sum))
