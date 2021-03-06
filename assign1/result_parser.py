import numpy as np
import pickle


class ResultParser(object):
    def __init__(self, outdir, num_of_episodes, num_of_steps):
        self.outdir = outdir
        self.num_of_episodes = num_of_episodes
        self.num_of_steps = num_of_steps

    def load_episode(self, episode_num):
        filename = '{0}/episode{1}'.format(self.outdir, episode_num)
        with open(filename, 'rb') as f:
            episode = pickle.load(f)
        return episode

    def get_episode_stats(self, episode_num):
        steps = self.load_episode(episode_num)
        env_states, agent_states = zip(*steps)
        env_states = list(env_states)
        agent_states = list(agent_states)
        rewards = [agent_state['reward'] for agent_state in agent_states]
        optimal_action = [env_state['optimal_action'] for env_state in env_states]
        action_taken = [agent_state['action'] for agent_state in agent_states]
        return (rewards, optimal_action, action_taken)

    def get_episode_aggregate_results(self):
        episode_stats = [self.get_episode_stats(i) for i in range(self.num_of_episodes)]
        episode_rewards = [episode_stat[0] for episode_stat in episode_stats]
        mean_reward = np.mean(np.array(episode_rewards), axis=0)
        episodes_optimal_actions = []
        for episode_num in range(self.num_of_episodes):
            _, opt_act, act_taken = episode_stats[episode_num]
            episode_optimal_actions = [float(a == b) for (a, b) in zip(opt_act, act_taken)]
            episodes_optimal_actions.append(episode_optimal_actions)
        mean_optimal_actions = np.mean(np.array(episodes_optimal_actions), axis=0)
        return [mean_reward, mean_optimal_actions]
