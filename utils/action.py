import numpy as np


def e_greedy(e, action_space, q_vector):
    if np.random.rand(1) < e:
        action = action_space.sample()
    else:
        action = np.argmax(q_vector)
    return action


def noisy_action(q_vector, decay_rate):
    noise_vector = np.random.randn(*q_vector.shape) * decay_rate
    action = np.argmax(q_vector + noise_vector)
    return action

