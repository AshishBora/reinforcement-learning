"""Windy Grid World class"""
import numpy as np


class WindyGridWorld(object):
    """Windy Grid World Environment"""
    def __init__(self, height, width, goal_state, wind_vec):
        self.height = height
        self.width = width
        self.goal_state = goal_state
        self.wind_vec = wind_vec
        self.action_space = ['LEFT', 'RIGHT', 'UP', 'DOWN']
        self.cur_state = None
        self._random_reset()

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
        if action == 'RIGHT':
            x_pos = x_pos + 1
        elif action == 'LEFT':
            x_pos = x_pos - 1
        elif action == 'UP':
            y_pos = y_pos + 1
        elif action == 'DOWN':
            y_pos = y_pos - 1
        return (x_pos, y_pos)

    def apply_wind(self, x_pos, y_pos):
        """Apply the effect of wind."""
        y_pos = y_pos + self.wind_vec[x_pos]
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
        return self.cur_state, reward, done, {}


class AlwaysXAgent(object):
    """Agent that always takes action X."""
    def __init__(self, action_space, always_action):
        self.action_space = action_space
        self.always_action = always_action
        self.prev_state = None
        self.prev_action = None

    def act(self, cur_state):
        """Get the action to be taken in the current state"""
        self.prev_state = cur_state
        action = self.always_action
        self.prev_action = action
        return action

    def update(self, reward):
        """Update any internal variables using the reward"""
        pass


def main():
    """Create grid world and call steps on it."""
    height = 7
    width = 10
    goal_state = (7, 3)
    wind_vec = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    grid_world = WindyGridWorld(height, width, goal_state, wind_vec)

    start_state = (0, 3)
    grid_world.state_reset(start_state)
    always_right_agent = AlwaysXAgent(grid_world.action_space, 'RIGHT')
    max_steps = 100
    done = False
    state = grid_world.cur_state
    print 0, state
    for step in range(max_steps):
        if not done:
            action = always_right_agent.act(state)
            state, reward, done, _ = grid_world.step(action)
            always_right_agent.update(reward)
            print step+1, state, reward, done


if __name__ == '__main__':
    main()
