import numpy as np


def evaulate(env, model, num_steps=1000):
    episode_rewards=[0.0]
    obs = env.reset()
    for i in range(num_steps):
        action = model.predict(obs)
        obs, reward, done, info = env.step(action)

        episode_rewards[-1] += reward

        if done:
            obs = env.reset()
            episode_rewards.append(0.0)

    mean = np.round(np.mean(episode_rewards[-100:]), 1)
    return mean
