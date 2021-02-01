import gym
import matplotlib.pyplot as plt

env = gym.make('LunarLanderContinuous-v2')
env.reset()
cords = []
for t in range(1000):
    action = env.action_space.sample()
    print('Action: ', action)
    observation, reward, done, info = env.step(action)
    cords.append((observation[0], observation[1], t))
    print('Observation: ', observation, ' Reward: ', reward)
    if done:
        print(f'Episode finished after {t+1} timesteps')
        break
env.close()



xs = [cord[0] for cord in cords]
ys = [cord[1] for cord in cords]
zs = [cord[2] for cord in cords]

plt.scatter(xs, ys, zs=zs)
plt.show()
