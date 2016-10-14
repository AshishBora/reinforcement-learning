"""Grid World class"""
import numpy as np


class GridWorld(object):
    """Grid World Environment"""
    def __init__(self, height, width, goal_state):
        self.height = height
        self.width = width
        self.goal_state = goal_state
        #         self.wind_loc = wind_loc
        #         self.wind_speed = wind_speed
        self.action_space = ['LEFT', 'RIGHT', 'UP', 'DOWN']
        self.cur_state = None
        self._reset()

    def _reset(self):
        x_pos = np.random.randint(self.width+1)
        y_pos = np.random.randint(self.height+1)
        self.cur_state = (x_pos, y_pos)

    def step(self, action):
        """Perform action in the current state"""
        assert action in self.action_space
        x_pos, y_pos = self.cur_state
        if action == 'RIGHT':
            x_pos = min(self.width-1, self.cur_state[0] + 1)
        elif action == 'LEFT':
            x_pos = max(0, self.cur_state[0] - 1)
        elif action == 'UP':
            y_pos = min(self.height-1, self.cur_state[1] + 1)
        elif action == 'DOWN':
            y_pos = max(0, self.cur_state[1] - 1)
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
    grid_world = GridWorld(3, 4, (2, 3))
    always_right_agent = AlwaysXAgent(grid_world.action_space, 'RIGHT')
    max_steps = 100
    state = grid_world.cur_state
    done = False
    for step in range(max_steps):
        if not done:
            action = always_right_agent.act(state)
            state, reward, done, _ = grid_world.step(action)
            always_right_agent.update(reward)
            print step, state, reward, done


if __name__ == '__main__':
    main()
