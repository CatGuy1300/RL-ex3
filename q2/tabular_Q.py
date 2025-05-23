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

    current_observation = s
    current_action = 0 # just for init
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        # TODO: Implement Q-Learning
        # 1. Choose an action by greedily (with noise) picking from Q table
        best_action = np.argmax(Q[s])
        epsilon = 0.01
        next_action_distribution = np.zeros(env.action_space.n) + (1 - epsilon) / env.action_space.n
        next_action_distribution[best_action] += epsilon
        new_action = np.random.choice(4, p=next_action_distribution)
        # 2. Get new state and reward from environment
        next_state, reward, terminated, truncated = env.step(new_action)
        # 3. Update Q-Table with new knowledge
        gamma_t = reward + y * np.max(Q[next_state]) - Q[s][new_action]
        Q[s][new_action] += + lr * gamma_t
        # 4. Update total reward
        rAll += reward
        # 5. Update episode if we reached the Goal State
        if terminated:
            break

        s = next_state

    
    rList.append(rAll)

# Reports
print("Score over time: " +  str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)
