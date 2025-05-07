#
# The Process of Q-learning
#
# \f[Q(s, a) ← Q(s, a) + \alpha [r + \gamma \max(Q(s', a')) - Q(s, a)]\f]
#
import numpy as np
from collections import defaultdict

# Define the Markov Game Environment for N agents
class MarkovGame:
    def __init__(self, num_agents, grid_size=3):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.actions = ['up', 'down', 'left', 'right']
        self.action_space = len(self.actions)
        self.state = tuple((np.random.randint(0, grid_size), np.random.randint(0, grid_size))
                          for _ in range(num_agents))
        self.goal = (grid_size-1, grid_size-1)  # Goal position for simplicity

    def step(self, actions):
        """
        Actions: list of actions, one per agent
        Returns: new_state, rewards, done
        """
        new_state = []
        rewards = [0] * self.num_agents

        # Update each agent's position
        for i, (pos, action) in enumerate(zip(self.state, actions)):
            new_pos = self._move(pos, action)
            new_state.append(new_pos)

            # Reward: +1 if agent reaches goal, -1 if collision with another agent
            if new_pos == self.goal:
                rewards[i] = 1
            for j, other_pos in enumerate(new_state):
                if i != j and new_pos == other_pos:
                    rewards[i] = -1
                    rewards[j] = -1

        self.state = tuple(new_state)
        done = any(r == 1 for r in rewards)  # End if any agent reaches goal
        return self.state, rewards, done

    def _move(self, pos, action):
        x, y = pos
        if   action == 'up': y = max(0, y-1)
        elif action == 'down': y = min(self.grid_size-1, y+1)
        elif action == 'left': x = max(0, x-1)
        elif action == 'right': x = min(self.grid_size-1, x+1)
        return (x, y)

    def reset(self):
        self.state = tuple((np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
                          for _ in range(self.num_agents))
        return self.state

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = defaultdict(lambda: np.zeros(4))  # 4 actions
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

# Training Loop for N Agents
def train_markov_game(num_agents, episodes=1000):
    env = MarkovGame(num_agents=num_agents)
    agents = [QLearningAgent() for _ in range(num_agents)]

    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            # Get actions for all agents
            actions = [agent.get_action(str(state[i]))
                      for i, agent in enumerate(agents)]

            # Step the environment with all actions
            next_state, rewards, done = env.step([env.actions[a] for a in actions])

            # Update each agent's Q-table
            for i, agent in enumerate(agents):
                agent.update(str(state[i]), actions[i], rewards[i], str(next_state[i]))

            state = next_state

        if episode % 100 == 0:
            print(f"Episode {episode} completed with {num_agents} agents")

    return env, agents

# Test the trained agents
def test_markov_game(env, agents):
    state = env.reset()
    done = False
    print(f"Initial state: {state}")
    while not done:
        actions = [agent.get_action(str(state[i]))
                  for i, agent in enumerate(agents)]
        state, rewards, done = env.step([env.actions[a] for a in actions])
        print(f"State: {state}, Rewards: {rewards}")

# Run with an arbitrary number of agents
if __name__=="main":
    num_agents = 3  # Change this to any number
    env, agents = train_markov_game(num_agents=num_agents, episodes=1000)
    test_markov_game(env, agents)
