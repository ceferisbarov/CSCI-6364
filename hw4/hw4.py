import gymnasium as gym
import numpy as np
from tqdm import tqdm
####### VVV Don't modify this part VVV #######

np.random.seed(1)

observation_grid_size = [30, 24, 30, 30]
observation_grid_middle = np.array([g/2 for g in observation_grid_size])
observation_discretization_width = np.array([0.16, 0.15, 0.014, 0.05])

def discretize_state(s):
    idxs = (s/observation_discretization_width + observation_grid_middle).astype(np.int32)
    for i in range(4):
        idxs[i] = max(min(idxs[i], observation_grid_size[i]-1), 0)
    return tuple(idxs)

def simulate_and_render(q_table, n_reps):
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(env, f'videos', episode_trigger=lambda x: True)
    for rep in range(n_reps):
        state = discretize_state(env.reset()[0])
        done = False
        rewards = 0.
        episode_step = 0
        while not done:
            episode_step += 1
            action = np.argmax(q_table[state])
            state, reward, done, _, _ = env.step(action)
            state = discretize_state(state)
            rewards += reward
            done = done or (episode_step >= 1000)
        print(f'Total reward for try #{rep}: {rewards}')
    env.close()

env = gym.make("CartPole-v1")
q_table = np.random.uniform(low=0, high=1, size=(observation_grid_size + [env.action_space.n]))

####### ^^^ Don't modify this part ^^^ #######

####### VVV Set hyperparameters VVV #######
n_episodes = 30000  # Maximum allowed episodes
gamma = 1        # Discount factor
epsilon = 0.98       # Initial exploration rate
lr = 0.2        # Learning rate
####### ^^^ Set hyperparameters ^^^ #######

####### Your code goes below #######

def train_q_learning():
    global epsilon

    for episode in tqdm(range(n_episodes)):
        epsilon = max(0.01, 1.0 - episode / (n_episodes * 0.8))

        continuous_state, _ = env.reset()
        discrete_state = discretize_state(continuous_state)
        done = False
        
        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[discrete_state])

            new_continuous_state, reward, terminated, _, _ = env.step(action)
            new_discrete_state = discretize_state(new_continuous_state)

            max_next_q = np.max(q_table[new_discrete_state])
            q_table[discrete_state + (action,)] = (1 - lr) * q_table[discrete_state + (action,)] + \
                                                   lr * (reward + gamma * max_next_q)

            discrete_state = new_discrete_state
            done = terminated or discrete_state[0] == 0 or discrete_state[0] == 29

            if done:
                break

    return q_table

trained_q_table = train_q_learning()

simulate_and_render(trained_q_table, n_reps=3)

env.close()
