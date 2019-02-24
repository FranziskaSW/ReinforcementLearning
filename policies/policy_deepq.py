from policies import base_policy as bp
from policies import DQNetwork as DQN
import numpy as np
from keras.models import Sequential
from keras.layers import *
import keras


EPSILON = 0.3
ALPHA = 0.5
GAMMA = 0.5
DROPOUT_RATE = 0.1
INPUT_SHAPE = (9, 9, 1)
NUM_ACTIONS = 3
BATCH_SIZE = 4

class MyPolicy(bp.Policy):
    """
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['batch_size'] = float(policy_args['batch_size']) if 'batch_size' in policy_args else BATCH_SIZE
        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.Q = DQN.DQNetwork(input_shape=(9, 9, 1), alpha=0.5, gamma=0.5,
                               dropout_rate=0.1, num_actions=3, batch_size=self.batch_size)
        self.memory = []
        self.act2idx = {'L': 0, 'R':1, 'F':2}


    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        if round >= 50:
           random_batches = np.random.choice(self.memory, self.batch_size)
           self.Q.learn(random_batches)

        try:
            if round % 100 == 0:
                if round > self.game_duration - self.score_scope:
                    self.log("Rewards in last 100 rounds which counts towards the score: " + str(self.r_sum), 'VALUE')
                else:
                    self.log("Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')
                self.r_sum = 0
            else:
                self.r_sum += reward

        except Exception as e:
            self.log("Something Went Wrong...", 'EXCEPTION')
            self.log(e, 'EXCEPTION')

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        board, head = new_state
        # head_pos, direction = head

        if round >=2:  # update to memory from previous round (prev_state)
            board_prev, head_prev = prev_state
            state_before = board_prev
            self.memory.append({'s_t': state_before, 'a_t': prev_action, 'r_t': reward, 's_tp1': board})

        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        else:
            q_values = self.Q.predict(board)
            idx = np.argmax(q_values)
            return bp.Policy.ACTIONS[idx]  # TODO: right now assume L, R, F as it is defined