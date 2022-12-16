from base import BaseAgent, Action
import numpy as np
from utils import get_action
from numpy import asarray
from numpy import savetxt


def init_q(s, a, init_type="ones"):
    if init_type == "ones":
        return np.ones((s, a))
    elif init_type == "random":
        return np.random.random((s, a))
    elif init_type == "zeros":
        return np.zeros((s, a))


def epsilon_greedy(q_table, epsilon, num_actions, s, train=False):
    if train or np.random.rand() < epsilon:
        print('argmax')
        return np.argmax(q_table[s, :])
    else:
        print('random')
        return np.random.randint(0, num_actions)


class Agent(BaseAgent):
    def __init__(self, alpha=0.01, gamma=0.999, epsilon=0.9):
        super(Agent, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state = 15
        self.n_actions = 9
        self.n_states = None
        self.action = None
        self.total_reward = 0
        self.last_score = 0
        self.reward_history = []
        self.q_table = None
        self.n_gems = 0
        self.gems_locations = []
        self.done = False

    def get_gems_locations(self):
        gems = []
        for x in range(self.grid_height):
            for y in range(self.grid_width):
                if self.grid[x][y] in ['1', '2', '3', '4']:
                    gems.append((x, y))
        return gems

    def observation_space(self):
        return self.grid_height * self.grid_width * (2 ** len(self.gems_locations))

    def get_state_from_pos(self, pos):
        return pos[0] * self.grid_width + pos[1]

    def get_agent_location(self):
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                if 'A' in self.grid[r][c]:
                    return self.get_state_from_pos((r, c))

    def percept(self, s, a, s_, reward):
        a_ = np.argmax(self.q_table[s_, :])
        if self.done:
            self.q_table[s, a] += self.alpha * (reward - self.q_table[s, a])
        else:
            self.q_table[s, a] += self.alpha * (reward + (self.gamma * self.q_table[s_, a_]) - self.q_table[s, a])
        self.state, self.action = s_, a_

    def reset(self):
        self.state = 15
        self.total_reward = 0
        self.last_score = 0
        self.done = False

    def step(self):
        gems_state = ''
        reward = self.agent_scores[0] - self.last_score
        agent_location = self.get_agent_location()
        new_gems_locations = self.get_gems_locations()
        for gem in self.gems_locations:
            if gem in new_gems_locations:
                gems_state += '1'
            else:
                gems_state += '0'
        s_ = agent_location * (2 ** self.n_gems) + int(gems_state, 2)
        if len(new_gems_locations) == 0 or self.max_turn_count == self.turn_count:
            self.done = True
        return s_, reward

    def do_turn(self) -> Action:
        # print(f'-------------------turn count {self.turn_count}---------------------')
        if self.turn_count == 1:
            self.reset()
            # print(f'reset result: state {self.state} total reward {self.total_reward}')
            self.action = epsilon_greedy(self.q_table, self.epsilon, self.n_actions, self.state)
            # print(f'q table {self.q_table}, action {self.action}')
        else:
            s_, reward = self.step()
            # print(f's_ {s_}, reward {reward}')
            self.total_reward += reward
            self.percept(self.state, self.action, s_, reward)
            # print(f'state {self.state}, action {self.action}')
            self.last_score = self.agent_scores[0]
            if self.done:
                print(f'total reward {self.total_reward}')
                self.reward_history.append(self.total_reward)
                # savetxt('reward_history1.csv', asarray([self.reward_history]), delimiter=',')
        return get_action(self.action)


class TestAgent(BaseAgent):
    def __init__(self, alpha=0.01, gamma=0.999, epsilon=0.9):
        super(TestAgent, self).__init__()
        self.state = 15
        self.n_actions = 9
        self.n_states = None
        self.action = None
        self.total_reward = 0
        self.last_score = 0
        self.epsilon = None
        self.reward_history = []
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.loadtxt('q_table1.txt')
        self.n_gems = 0
        self.gems_locations = []

    def get_gems_locations(self):
        gems = []
        for x in range(self.grid_height):
            for y in range(self.grid_width):
                if self.grid[x][y] in ['1', '2', '3', '4']:
                    gems.append((x, y))
        return gems

    def reset(self):
        self.state = 15
        self.total_reward = 0
        self.last_score = 0

    def observation_space(self):
        n_gems = 0
        for x in range(self.grid_height):
            for y in range(self.grid_width):
                if self.grid[x][y] in ['1', '2', '3', '4']:
                    n_gems += 1
        return self.grid_height * self.grid_width * (2 ** n_gems), n_gems

    def get_state_from_pos(self, pos):
        return pos[0] * self.grid_width + pos[1]

    def get_pos_from_state(self, state):
        return state // self.grid_width, state % self.grid_width

    def get_agent_location(self):
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                if 'A' in self.grid[r][c]:
                    return self.get_state_from_pos((r, c))

    def step(self):
        done = False
        gems_state = ''
        reward = self.agent_scores[0] - self.last_score
        print(f'rewrad {reward}')
        agent_location = self.get_agent_location()
        print(f'agent loc {agent_location}')
        new_gems_locations = self.get_gems_locations()
        print(f'new gems loc {new_gems_locations}')
        for gem in self.gems_locations:
            if gem in new_gems_locations:
                gems_state += '1'
            else:
                gems_state += '0'
        print(gems_state, int(gems_state, 2))
        s_ = agent_location * (2 ** self.n_gems) + int(gems_state, 2)
        print(f's_ {s_}')
        if len(new_gems_locations) == 0 or self.max_turn_count == self.turn_count:
            print(f"WON")
            done = True
        return s_, reward, done

    def do_turn(self) -> Action:
        if self.turn_count != 1:
            print(f'-------------------------turn count {self.turn_count}---------------------------------------')
            self.state, r, d = self.step()
            self.action = epsilon_greedy(self.q_table, 0, 9, self.state, train=True)
            print(f"Chose action {self.action} for state {self.state}")
            self.last_score = self.agent_scores[0]
        return get_action(self.action)


if __name__ == '__main__':
    agent = Agent()
    agent.grid = np.loadtxt('test.txt', str)
    agent.gems_locations = agent.get_gems_locations()
    agent.n_states = agent.observation_space()
    agent.n_gems = len(agent.gems_locations)
    agent.q_table = init_q(agent.n_states, agent.n_actions, init_type="ones")
    data = agent.play()
    savetxt('reward_history.csv', asarray([agent.reward_history]), delimiter=',')
    np.savetxt('q_table.txt', agent.q_table)

    # test_agent = TestAgent()
    # test_agent.grid = np.loadtxt('test.txt', str)
    # test_agent.gems_locations = test_agent.get_gems_locations()
    # test_agent.reset()
    # test_agent.epsilon = 0
    # test_agent.action = epsilon_greedy(test_agent.q_table, test_agent.epsilon, test_agent.n_actions, test_agent.state, train=True)
    # data = test_agent.play()
    print("FINISH : ", data)
