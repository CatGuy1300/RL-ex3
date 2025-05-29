from importlib.metadata import distribution

import gym
import numpy as np

# Load environment
env = gym.make('FrozenLake-v0')

# Implement Q-Table learning algorithm
#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .8
y = .95
epsilon = 0.1
num_episodes = 2000
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    rAll = 0 # Total reward during current episode
    d = False
    j = 0

    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        # 1. Choose an action by greedily (with noise) picking from Q table
        action = np.argmax(Q[s] + np.random.normal(0, 0.01, size=env.action_space.n))
        # 2. Get new state and reward from environment
        next_state, reward, done, info = env.step(action)
        # 3. Update Q-Table with new knowledge
        gamma_t = reward + y * np.max(Q[next_state]) - Q[s][action]
        Q[s][action] += lr * gamma_t
        # 4. Update total reward
        rAll += reward
        # 5. Update episode if we reached the Goal State
        if done:
            break

        s = next_state

    
    rList.append(rAll)

# Reports
print("Score over time: " +  str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)
