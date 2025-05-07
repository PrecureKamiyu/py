import numpy as np
from collections import defaultdict

# Simplified Car Network Environment
class CarNetworkGame:
    def __init__(self, num_cars, grid_size=5):
        self.num_cars = num_cars
        self.grid_size = grid_size
        self.actions = ['up', 'down', 'left', 'right', 'stay']
        self.action_space = len(self.actions)
        self.state = tuple((np.random.randint(0, grid_size), np.random.randint(0, grid_size))
                          for _ in range(num_cars))
        # Dynamic goal (e.g., based on network demand)
        self.goal = (grid_size-1, grid_size-1)

    def step(self, actions):
        new_state = []
        rewards = [0] * self.num_cars

        for i, (pos, action) in enumerate(zip(self.state, actions)):
            new_pos = self._move(pos, action)
            new_state.append(new_pos)
            # Reward: +1 for reaching goal, -0.1 for collision, -0.01 for moving
            if new_pos == self.goal:
                rewards[i] = 1
            for j, other_pos in enumerate(new_state):
                if i != j and new_pos == other_pos:
                    rewards[i] -= 0.1
            if action != 'stay':
                rewards[i] -= 0.01

        self.state = tuple(new_state)
        # Update goal dynamically (e.g., based on real-time network data)
        self._update_goal()
        return self.state, rewards

    def _move(self, pos, action):
        x, y = pos
        if action == 'up': y = max(0, y-1)
        elif action == 'down': y = min(self.grid_size-1, y+1)
        elif action == 'left': x = max(0, x-1)
        elif action == 'right': x = min(self.grid_size-1, x+1)
        return (x, y)

    def _update_goal(self):
        # Simulate dynamic network requirement (e.g., new destination)
        if np.random.random() < 0.1:  # 10% chance to change goal
            self.goal = (np.random.randint(0, self.grid_size),
                        np.random.randint(0, self.grid_size))

    def reset(self):
        self.state = tuple((np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
                          for _ in range(self.num_cars))
        return self.state

# Online Joint Q-Learning Agent
class OnlineJointQLearningAgent:
    def __init__(self, num_cars, num_actions=5, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.num_cars = num_cars
        self.num_actions = num_actions
        self.q_table = defaultdict(lambda: np.zeros(num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_action(self, joint_state, agent_idx):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        return np.argmax(self.q_table[joint_state])

    def update(self, joint_state, joint_action, reward, next_joint_state, agent_idx):
        old_q = self.q_table[joint_state][joint_action[agent_idx]]
        next_max_q = np.max(self.q_table[next_joint_state])
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        self.q_table[joint_state][joint_action[agent_idx]] = new_q

# Online Decision-Making Loop
def run_online_car_network(num_cars, timesteps=1000):
    env = CarNetworkGame(num_cars=num_cars)
    agents = [OnlineJointQLearningAgent(num_cars=num_cars) for _ in range(num_cars)]

    joint_state = env.reset()
    for t in range(timesteps):
        # Real-time decision
        joint_action = [agent.get_action(joint_state, i)
                       for i, agent in enumerate(agents)]
        next_joint_state, rewards = env.step([env.actions[a] for a in joint_action])

        # Incremental update
        for i, agent in enumerate(agents):
            agent.update(joint_state, joint_action, rewards[i], next_joint_state, i)

        joint_state = next_joint_state

        if t % 100 == 0:
            print(f"Timestep {t}: State: {joint_state}, Rewards: {rewards}")

# Run with 3 cars
run_online_car_network(num_cars=3, timesteps=1000)
