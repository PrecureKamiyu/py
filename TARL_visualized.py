import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the Markov Game Environment
class SimpleMarkovGame:
    def __init__(self):
        self.grid_size = 3
        self.state = ((0, 0), (2, 2))  # (agent1_pos, agent2_pos)
        self.actions = ['up', 'down', 'left', 'right']
        self.action_space = len(self.actions)

    def step(self, action1, action2):
        pos1, pos2 = self.state
        new_pos1 = self._move(pos1, action1)
        new_pos2 = self._move(pos2, action2)

        reward1 = 1 if new_pos1 == (2, 2) else 0
        reward2 = -1 if new_pos2 == new_pos1 else 0

        self.state = (new_pos1, new_pos2)
        done = reward1 == 1 or reward2 == -1
        return self.state, (reward1, reward2), done, (action1, action2)

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
        self.q_table = defaultdict(lambda: np.zeros(4))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 4)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        old_q = self.q_table[state][action]
        next_max_q = np.max(self.q_table[next_state])
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        self.q_table[state][action] = new_q

# Visualization function
def visualize_game(history):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.grid(True)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))

    # Plot target position
    ax.plot(2, 2, 'g*', markersize=15, label='Target (2,2)')

    # Initial positions
    agent1_plot, = ax.plot([], [], 'bo', markersize=15, label='Agent 1')
    agent2_plot, = ax.plot([], [], 'ro', markersize=15, label='Agent 2')

    # Text annotations for actions
    action1_text = ax.text(0, 0, '', ha='center', va='center')
    action2_text = ax.text(0, 0, '', ha='center', va='center')

    def update(frame):
        state, (action1, action2) = history[frame]
        x1, y1 = state[0]
        x2, y2 = state[1]

        agent1_plot.set_data([x1], [y1])
        agent2_plot.set_data([x2], [y2])

        action1_text.set_position((x1, y1 + 0.2))
        action1_text.set_text(action1)
        action2_text.set_position((x2, y2 + 0.2))
        action2_text.set_text(action2)

        return agent1_plot, agent2_plot, action1_text, action2_text

    ax.legend()
    ani = animation.FuncAnimation(fig, update, frames=len(history),
                                 interval=1000, blit=True, repeat=False)
    plt.title("Two Agent Markov Game")
    plt.show()

# Training and Recording
env = SimpleMarkovGame()
agent1 = QLearningAgent()
agent2 = QLearningAgent()

# Training phase
for episode in range(1000):
    state = env.reset()
    done = False
    while not done:
        action1 = agent1.get_action(str(state[0]))
        action2 = agent2.get_action(str(state[1]))

        next_state, (reward1, reward2), done, _ = env.step(
            env.actions[action1], env.actions[action2]
        )

        agent1.update(str(state[0]), action1, reward1, str(next_state[0]))
        agent2.update(str(state[1]), action2, reward2, str(next_state[1]))
        state = next_state

    if episode % 100 == 0:
        print(f"Episode {episode} completed")

# Testing phase with recording
state = env.reset()
done = False
history = [(state, ("start", "start"))]  # Record initial state

while not done:
    action1 = agent1.get_action(str(state[0]))
    action2 = agent2.get_action(str(state[1]))
    next_state, rewards, done, actions = env.step(env.actions[action1], env.actions[action2])
    history.append((next_state, (env.actions[action1], env.actions[action2])))
    print(f"State: {next_state}, Actions: {env.actions[action1]}, {env.actions[action2]}, Rewards: {rewards}")
    state = next_state

# Visualize the results
visualize_game(history)
