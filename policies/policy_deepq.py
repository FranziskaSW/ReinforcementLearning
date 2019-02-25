from policies import base_policy as bp
from policies import DQNetwork as DQN
import numpy as np
from keras.models import Sequential
from keras.layers import *
import keras
import os
import pickle

global cwd
cwd = os.getcwd()


EPSILON = 0.3
ALPHA = 0.5
GAMMA = 0.5
DROPOUT_RATE = 0.1

NUM_ACTIONS = 3  # (L, R, F)
BATCH_SIZE = 32
VICINITY = 4
INPUT_SHAPE = (VICINITY*2+1, VICINITY*2+1, 1)

class MyPolicy(bp.Policy):
    """
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['batch_size'] = float(policy_args['batch_size']) if 'batch_size' in policy_args else BATCH_SIZE
        policy_args['vicinity'] = float(policy_args['vicinity']) if 'vicinity' in policy_args else VICINITY
        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.Q = DQN.DQNetwork(input_shape=(9, 9, 1), alpha=0.5, gamma=0.5,
                               dropout_rate=0.1, num_actions=NUM_ACTIONS, batch_size=self.batch_size)
        self.memory = []
        self.losses = []
        self.act2idx = {'L': 0, 'R': 1, 'F': 2}
        self.idx2act = {0: 'L', 1: 'R', 2: 'F'}

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        if round >= 50:
            random_batches = np.random.choice(self.memory, self.batch_size)
            loss = self.Q.learn(random_batches)
            self.losses.append(loss)

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

    def getVicinityMap(self, board, center, direction):
        vicinity = self.vicinity

        r, c = center
        left = c - vicinity
        right = c + vicinity + 1
        top = r - vicinity
        bottom = r + vicinity + 1

        big_board = board

        if left < 0:
            left_patch = np.matrix(big_board[:, left])
            left = 0
            right = 2 * vicinity + 1
            big_board = np.hstack([left_patch.T, big_board])

        if right >= board.shape[1]:
            right_patch = np.matrix(big_board[:, :(right % board.shape[1] + 1)])
            big_board = np.hstack([big_board, right_patch])

        if top < 0:
            top_patch = np.matrix(big_board[top, :])
            top = 0
            bottom = 2 * vicinity + 1
            big_board = np.vstack([top_patch, big_board])

        if bottom >= board.shape[0]:
            bottom_patch = np.matrix(big_board[:(bottom % board.shape[0])])
            big_board = np.vstack([big_board, bottom_patch])

        map = big_board[top:bottom, left:right]

        if direction == 'N': return map
        if direction == 'E': return np.rot90(map, k=1)
        if direction == 'S': return np.rot90(map, k=2)
        if direction == 'W': return np.rot90(map, k=-1)

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        board, head = new_state
        head_pos, direction = head
        map_new = self.getVicinityMap(board, head_pos, direction)

        if round >=2:  # update to memory from previous round (prev_state)
            board_prev, head_prev = prev_state
            head_pos_prev, direction_prev = head_prev
            map_before = self.getVicinityMap(board_prev, head_pos_prev, direction_prev)
            self.memory.append({'s_t': map_before, 'a_t': prev_action, 'r_t': reward, 's_tp1': map_new})

        if round == 2000:
            losses = self.losses
            with open(cwd + "/losses.pickle", "wb") as f:
                pickle.dump(losses, f)

        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        else:
            q_values = self.Q.predict(map_new)
            idx = np.argmax(q_values)
            return self.idx2act[idx]  # TODO: right now assume L, R, F as it is defined