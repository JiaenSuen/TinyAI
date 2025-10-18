import time
import numpy as np
from collections import defaultdict

from Maze import MazeEnv



env = MazeEnv()#map_file="mazes/_maze.txt"
env.reset()


# --- Q-Learning Reinforcement Learning  ----




# Build Q-table
Q = defaultdict(lambda: np.zeros(len(MazeEnv.actions)))


# Parameters
alpha = 0.1     # learning rate
gamma = 0.95    # discount factor
epsilon = 0.3   # epsilon-greedy
episodes = 5000 # iteration times
max_steps = 400 # max Step for each Eppch




for ep in range(episodes):
    state = env.reset()
    done  = False

    for _ in range(max_steps):
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(len(env.actions))
        else:
            action_idx = np.argmax(Q[state])

        action = env.actions[action_idx]
        next_state, reward, done = env.step(action)

        # Q-learning Transfer Function
        Q[state][action_idx] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state][action_idx]
        )

        state = next_state

        if done:
            break





state = env.reset()
done = False
path = []  # Record

while not done:
    action_idx = np.argmax(Q[state])
    action = env.actions[action_idx]
    path.append(action)
    next_state, reward, done = env.step(action)
    state = next_state
    if len(path) > 100:
        break  

print("Final Path:", path)


env.play(path, delay=0.3, save_path="Record/q_learning_maze.gif", show=False)
