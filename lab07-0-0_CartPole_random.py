import gym


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.reset()
    random_episodes = 0
    reward_sum = 0
    rList = []

    while random_episodes < 10:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        print(observation, reward, done)

        reward_sum += reward

        if done:
            random_episodes += 1
            print('Reward for this episode was:', reward_sum)
            rList.append(reward_sum)
            reward_sum = 0
            env.reset()

    env.close()

    print('Reward sum of each episode:', rList)
    print('Maximum reward sum:', max(rList))
    print('Average reward sum:', sum(rList)/len(rList))

