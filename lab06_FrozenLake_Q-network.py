import gym
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.action import *


def one_hot(x):
    x_ = np.identity(16)[x:x+1]
    return x_.astype(np.float32)


@tf.function
def forward(x):
    return tf.matmul(x, W)


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        Qpred = forward(x)
        loss = tf.reduce_sum(tf.square(y - Qpred))

    gradients = tape.gradient(loss, W)
    optimizer.apply_gradients([(gradients, W)])


if __name__ == '__main__':

    env = gym.make('FrozenLake-v0')
    env.render()

    input_size = env.observation_space.n
    output_size = env.action_space.n
    learning_rate = 0.1

    W = tf.Variable(tf.random.uniform([input_size, output_size], 0, 0.01), dtype=tf.float32)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    dis = 0.99
    num_episodes = 2000

    rList = []

    for i in tqdm(range(num_episodes)):
        s = env.reset()
        e = .5 / ((i/100) + 1)
        rAll = 0
        done = False

        while not done:
            Qs = forward(one_hot(s)).numpy()

            # a = e_greedy(e, env.action_space, Qs)
            a = noisy_action(Qs, e)

            s1, reward, done, _ = env.step(a)
            rAll += reward

            if done:
                Qs[0, a] = reward
            else:
                Qs1 = forward(one_hot(s1)).numpy()
                Qs[0, a] = reward + dis * np.max(Qs1)

            train_step(one_hot(s), Qs)

            s = s1

        rList.append(rAll)

    print('Success rate:', str(sum(rList)/num_episodes))

    plt.bar(range(len(rList)), rList)
    plt.show()

