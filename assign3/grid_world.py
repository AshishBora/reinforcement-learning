"""Grid World class"""
import random
import numpy as np
import matplotlib.pyplot as plt


class StochasticWindyGridWorld(object):
    """Stochastic Windy Grid World Environment"""
    def __init__(self, height, width, goal_state, wind_vec, move_type, stoch_prob):
        self.height = height
        self.width = width
        self.goal_state = goal_state
        self.wind_vec = wind_vec
        self.stoch_prob = stoch_prob
        if move_type == 'ALL':
            self.action_space = ALL_MOVES
        if move_type == 'KING':
            self.action_space = ALL_MOVES[:4] + ALL_MOVES[5:]
        elif move_type == 'MANHATTAN':
            self.action_space = [ALL_MOVES[i] for i in [1, 3, 5, 7]]
        self.cur_state = None
        self._random_reset()

    def get_all_states(self):
        """Return all states."""
        all_states = []
        for x_pos in range(self.width):
            for y_pos in range(self.height):
                all_states.append((x_pos, y_pos))
        return all_states

    def _random_reset(self):
        x_pos = np.random.randint(self.width+1)
        y_pos = np.random.randint(self.height+1)
        self.cur_state = (x_pos, y_pos)

    def state_reset(self, start_state):
        """Reset state to a given start state."""
        self.cur_state = start_state

    def move(self, x_pos, y_pos, action):
        """Move accoding to the given action."""
        assert action in self.action_space
        delta_x, delta_y = map(int, action.split(','))
        x_pos += delta_x
        y_pos += delta_y
        return (x_pos, y_pos)

    def apply_wind(self, x_pos, y_pos):
        """Apply the effect of wind."""
        wind_movement = self.wind_vec[x_pos]
        p = [self.stoch_prob, 1 - 2*self.stoch_prob, self.stoch_prob]
        wind_movement += np.random.choice([-1, 0, 1], p=p)
        y_pos = y_pos + wind_movement
        return (x_pos, y_pos)

    def back2grid(self, x_pos, y_pos):
        """Get back to the grid."""
        x_pos = max(0, min(self.width - 1, x_pos))
        y_pos = max(0, min(self.height - 1, y_pos))
        return (x_pos, y_pos)

    def step(self, action):
        """Perform action in the current state"""
        x_pos, y_pos = self.cur_state
        x_pos, y_pos = self.apply_wind(x_pos, y_pos)
        x_pos, y_pos = self.back2grid(x_pos, y_pos)
        x_pos, y_pos = self.move(x_pos, y_pos, action)
        x_pos, y_pos = self.back2grid(x_pos, y_pos)
        self.cur_state = (x_pos, y_pos)
        reward = -1
        done = (self.cur_state == self.goal_state)
        return self.cur_state, reward, done


class AlwaysXAgent(object):
    """Agent that always takes action X."""
    def __init__(self, action_space, always_action):
        self.action_space = action_space
        self.always_action = always_action

    def act(self):
        """Get the action to be taken in the current state"""
        action = self.always_action
        return action

    def update(self, reward):
        """Update any internal variables using the reward"""
        pass


class SarsaAgent(object):
    """Agent that implements Sarsa"""
    def __init__(self, state_space, action_space, epsilon, alpha, gamma):
        self.state_space = state_space
        self.action_space = action_space
        self.Q = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        for state in self.state_space:
            self.Q[state] = {}
            for action in self.action_space:
                self.Q[state][action] = 0

    def eps_greedy(self, action_q_vals):
        """Epsilon greedy action selection."""
        if np.random.rand() < self.epsilon:
            action = random.choice(action_q_vals.keys())
        else:
            action = max(action_q_vals.items(), key=lambda(k, v): v)[0]
        return action

    def act(self, cur_state):
        """Get the action to be taken in the current state"""
        action_q_vals = self.Q[cur_state]
        action = self.eps_greedy(action_q_vals)
        return action

    def update(self, S, A, R, S_prime, A_prime):
        """Using the reward, update Q values"""
        self.Q[S][A] += self.alpha * (
            R + self.gamma*(self.Q[S_prime][A_prime]) - self.Q[S][A])


def main(move_type, stoch_prob):
    """Create windy grid world and use SARSA agent on it"""
    height = 7
    width = 10
    goal_state = (7, 3)
    wind_vec = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    grid_world = StochasticWindyGridWorld(height, width, goal_state, wind_vec, move_type, stoch_prob)
    start_state = (0, 3)
    epsilon, alpha, gamma = 0.1, 0.5, 1
    sarsa_agent = SarsaAgent(grid_world.get_all_states(),
                             grid_world.action_space,
                             epsilon, alpha, gamma)

    max_steps = 8000
    done = True
    count = -1
    counts = []
    for _ in range(max_steps+1):
        if done:
            count += 1
            done = False
            grid_world.state_reset(start_state)
            S = grid_world.cur_state
            A = sarsa_agent.act(S)

        S_prime, R, done = grid_world.step(A)
        A_prime = sarsa_agent.act(S_prime)
        sarsa_agent.update(S, A, R, S_prime, A_prime)
        S = S_prime
        A = A_prime
        counts.append(count)

    plt.plot(counts)
    plt.show()


if __name__ == '__main__':
    ALL_MOVES = ['-1, 1', '0, 1', '1, 1',
                 '-1, 0', '0, 0', '1, 0',
                 '-1,-1', '0,-1', '1,-1']
    MOVE_TYPE = 'MANHATTAN'
    STOCH_PROB = 0.33
    main(MOVE_TYPE, STOCH_PROB)
