import numpy as np


def evaulate(env, model, num_episodes=5, show_render=True):
    episode_rewards=[]
    obs = env.reset()
    for i in range(num_episodes):
        obs = env.reset()
        episode_rewards.append(0.0)
        done = False
        # steps = 1
        while not done:
            action = model.predict(obs)
            obs, reward, done, info = env.step(action)
            print(f"action chosen: {action}")
            # print(obs)
            if show_render: env.render()
            episode_rewards[-1] += reward

    mean = np.round(np.mean(episode_rewards),1)
    return mean
