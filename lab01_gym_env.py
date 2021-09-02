import gym


# PRINT ALL GAMES
def print_all_envs():
    all_envs = sorted(gym.envs.registry.all(), key=lambda x: x.id)
    for env_spec in all_envs:
        print(env_spec.id)


def show_env(id):
    env = gym.make(id)
    print(id)
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)
    print(env.metadata)


# SAMPLE GAMES
def main():
    env = gym.make("Taxi-v3")
    observation = env.reset()
    print(observation)

    for _ in range(20):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        print(action, observation, reward, done, info)


if __name__ == '__main__':
    print_all_envs()
    show_env("Breakout-v4")
    # main()

