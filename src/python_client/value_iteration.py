import numpy as np
import matplotlib.pyplot as plt


class ValueIteration:
    def __init__(self, reward_function, transition_model, grid, gamma):
        self.num_states = transition_model.shape[0]
        self.num_actions = transition_model.shape[1]
        self.reward_function = np.nan_to_num(reward_function)
        self.transition_model = transition_model
        self.gamma = gamma
        self.values = np.zeros(self.num_states)
        self.policy = None
        self.map = grid
        self.num_rows = grid.shape[0]
        self.num_cols = grid.shape[1]
        self.state_space = {}

    def get_state_from_pos(self, pos):
        return pos[0] * self.num_cols + pos[1]

    def get_pos_from_state(self, state):
        return state // self.num_cols, state % self.num_cols

    def one_iteration(self, state_space):
        delta = 0
        for s in state_space:
            temp = self.values[s]
            v_list = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                p = self.transition_model[s, a]
                if a in [0, 1, 2, 3]:
                    v_list[a] = self.reward_function[s] + self.gamma * np.sum(p * self.values)
                elif a in [4, 5, 6, 7]:
                    v_list[a] = 2 * self.reward_function[s] + self.gamma * np.sum(p * self.values)
                elif a == 8:
                    v_list[a] = self.gamma * np.sum(p * self.values)

            self.values[s] = max(v_list)
            delta = max(delta, abs(temp - self.values[s]))
        return delta

    def get_policy(self):
        pi = np.ones(self.num_states) * -1
        for s in range(self.num_states):
            v_list = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                p = self.transition_model[s, a]
                if a in [0, 1, 2, 3]:
                    v_list[a] = self.reward_function[s] + self.gamma * np.sum(p * self.values)
                elif a in [4, 5, 6, 7]:
                    v_list[a] = 2 * self.reward_function[s] + self.gamma * np.sum(p * self.values)
                elif a == 8:
                    v_list[a] = self.gamma * np.sum(p * self.values)

            max_index = []
            max_val = np.max(v_list)
            for a in range(self.num_actions):
                if v_list[a] == max_val:
                    max_index.append(a)
            if 8 in max_index:
                pi[s] = 8
            else:
                pi[s] = np.random.choice(max_index)

        return pi.astype(int)

    def get_state_space(self, state):
        new_map_states = set()
        r, c = self.get_pos_from_state(state)
        for x in [-2, -1, 0, 1, 2]:
            for y in [-2, -1, 0, 1, 2]:
                if x < 0:
                    if y < 0:
                        s = self.get_state_from_pos((max(r+x, 0), max(c+y, 0)))
                    else:
                        s = self.get_state_from_pos((max(r + x, 0), min(c + y, self.num_cols - 1)))
                else:
                    if y < 0:
                        s = self.get_state_from_pos((min(r + x, self.num_rows - 1), max(c+y, 0)))
                    else:
                        s = self.get_state_from_pos((min(r + x, self.num_rows - 1), min(c + y, self.num_cols - 1)))
                new_map_states.add(s)
        self.state_space[state] = list(new_map_states)

    def train(self, tol=1e-3, plot=True):
        for s in range(self.num_states):
            self.get_state_space(s)
        state = 0
        delta = self.one_iteration(self.state_space[state])
        delta_history = [delta]
        flag = False
        while True:
            state += 1
            if state == self.num_states:
                flag = True
                state = 0
            delta = self.one_iteration(self.state_space[state])
            delta_history.append(delta)
            if delta < tol and flag is True:
                break
        self.policy = self.get_policy()

        # if plot is True:
        #     fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=200)
        #     ax.plot(np.arange(len(delta_history)) + 1, delta_history, marker='o', markersize=4,
        #             alpha=0.7, color='#2ca02c', label=r'$\gamma= $' + f'{self.gamma}')
        #     ax.set_xlabel('Iteration')
        #     ax.set_ylabel('Delta')
        #     ax.legend()
        #     plt.tight_layout()
        #     plt.show()
