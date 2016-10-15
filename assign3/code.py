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
        p = [self.stoch_prob/2.0, 1 - self.stoch_prob, self.stoch_prob/2.0]
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


def plot_episodes(counts):
    plt.plot(counts)
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')


def get_expt1():
    model = {'MOVE_TYPE': 'MANHATTAN',
             'STOCH_PROB' : 0.0,
             'EPSILON': 0.1,
             'legend': ''}
    expt = [model]
    return expt


def get_expt2():
    model1 = {'MOVE_TYPE': 'MANHATTAN', 'STOCH_PROB' : 0.0}
    model2 = {'MOVE_TYPE': 'KING', 'STOCH_PROB' : 0.0}
    model3 = {'MOVE_TYPE': 'ALL', 'STOCH_PROB' : 0.0}
    expt = [model1, model2, model3]
    for model in expt:
        model['EPSILON'] = 0.1
        model['legend'] = model['MOVE_TYPE']
    return expt


def get_expt3():
    model1 = {'MOVE_TYPE': 'KING', 'STOCH_PROB': 0.0, 'EPSILON': 0.1}
    model2 = {'MOVE_TYPE': 'KING', 'STOCH_PROB': 0.0, 'EPSILON': 0.05}
    model3 = {'MOVE_TYPE': 'KING', 'STOCH_PROB': 0.0, 'EPSILON': 0.01}
    model4 = {'MOVE_TYPE': 'KING', 'STOCH_PROB': 0.0, 'EPSILON': 0.001}
    model5 = {'MOVE_TYPE': 'KING', 'STOCH_PROB': 0.0, 'EPSILON': 0.0}
    expt = [model1, model2, model3, model4, model5]
    for model in expt:
        model['legend'] =  model['MOVE_TYPE'] + ', EPSILON = ' + str(model['EPSILON'])
    return expt


def get_expt4():
    model1 = {'MOVE_TYPE': 'KING', 'STOCH_PROB': 1.0, 'EPSILON': 0.001}
    model2 = {'MOVE_TYPE': 'KING', 'STOCH_PROB': 0.5, 'EPSILON': 0.001}
    model3 = {'MOVE_TYPE': 'KING', 'STOCH_PROB': 0.2, 'EPSILON': 0.001}
    model4 = {'MOVE_TYPE': 'KING', 'STOCH_PROB': 0.1, 'EPSILON': 0.001}
    model5 = {'MOVE_TYPE': 'KING', 'STOCH_PROB': 0.05, 'EPSILON': 0.001}
    model6 = {'MOVE_TYPE': 'KING', 'STOCH_PROB': 0.0, 'EPSILON': 0.001}
    expt = [model1, model2, model3, model4, model5, model6]
    for model in expt:
        model['legend'] =  model['MOVE_TYPE'] + ', STOCH_PROB = ' + str(model['STOCH_PROB'])
    return expt


def main(move_type, stoch_prob, epsilon):
    """Create windy grid world and use SARSA agent on it"""
    height = 7
    width = 10
    goal_state = (7, 3)
    wind_vec = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    grid_world = StochasticWindyGridWorld(height, width, goal_state, wind_vec, move_type, stoch_prob)
    start_state = (0, 3)
    alpha, gamma = 0.5, 1
    sarsa_agent = SarsaAgent(grid_world.get_all_states(),
                             grid_world.action_space,
                             epsilon, alpha, gamma)
    max_steps = 10000
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
    return counts


if __name__ == '__main__':
    ALL_MOVES = ['-1, 1', '0, 1', '1, 1',
                 '-1, 0', '0, 0', '1, 0',
                 '-1,-1', '0,-1', '1,-1']
    expt = get_expt4()
    plt.figure(figsize=[10, 7])
    for model in expt:
        MOVE_TYPE = model['MOVE_TYPE']
        STOCH_PROB = model['STOCH_PROB']
        EPSILON = model['EPSILON']
        counts = main(MOVE_TYPE, STOCH_PROB, EPSILON)
        plot_episodes(counts)
    legends = [model['legend'] for model in expt]
    plt.legend(legends, loc='upper left')
    plt.show()
