# Two agent reinforcement learning
# using Markov game

import numpy as np
from collections import defaultdict

# Define the Markov Game Environment
class SimpleMarkovGame:
    def __init__(self):
        self.grid_size = 3
        self.state = ((0, 0), (2, 2))  # (agent1_pos, agent2_pos)
        self.actions = ['up', 'down', 'left', 'right']
        self.action_space = len(self.actions)

    def step(self, action1, action2):
        pos1, pos2 = self.state
        # Update positions based on actions
        new_pos1 = self._move(pos1, action1)
        new_pos2 = self._move(pos2, action2)

        # Reward: +1 if agent1 reaches (2,2), -1 if agent2 blocks
        reward1 = 1 if new_pos1 == (2, 2) else 0
        reward2 = -1 if new_pos2 == new_pos1 else 0

        self.state = (new_pos1, new_pos2)
        done = reward1 == 1 or reward2 == -1
        return self.state, (reward1, reward2), done

    def _move(self, pos, action):
        x, y = pos
        if action == 'up': y = max(0, y-1)
        elif action == 'down': y = min(self.grid_size-1, y+1)
        elif action == 'left': x = max(0, x-1)
        elif action == 'right': x = min(self.grid_size-1, x+1)
        return (x, y)

    def reset(self):
        self.state = ((0, 0), (2, 2))
        return self.state

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = defaultdict(lambda: np.zeros(4))  # 4 actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 4)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        old_q = self.q_table[state][action]
        next_max_q = np.max(self.q_table[next_state])
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        self.q_table[state][action] = new_q

# Training Loop
env = SimpleMarkovGame()
agent1 = QLearningAgent()
agent2 = QLearningAgent()

for episode in range(1000):
    state = env.reset()
    done = False
    while not done:
        action1 = agent1.get_action(str(state[0]))  # Agent 1's action
        action2 = agent2.get_action(str(state[1]))  # Agent 2's action

        next_state, (reward1, reward2), done = env.step(
            env.actions[action1], env.actions[action2]
        )

        # Update Q-tables
        agent1.update(str(state[0]), action1, reward1, str(next_state[0]))
        agent2.update(str(state[1]), action2, reward2, str(next_state[1]))

        state = next_state

    if episode % 100 == 0:
        print(f"Episode {episode} completed")

# Test the trained agents
state = env.reset()
done = False
while not done:
    action1 = agent1.get_action(str(state[0]))
    action2 = agent2.get_action(str(state[1]))
    state, rewards, done = env.step(env.actions[action1], env.actions[action2])
    print(f"State: {state}, Rewards: {rewards}")
