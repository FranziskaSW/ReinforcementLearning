from policies import base_policy as bp
from policies import DQNetwork as DQN
import numpy as np
from keras.models import Sequential
from keras.layers import *
import keras
import os
import pickle # TODO REMOVE

global cwd
cwd = os.getcwd()


EPSILON = 0.3
EPSILON_RATE = 0.999
GAMMA = 0.5
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.0001

NUM_ACTIONS = 3  # (L, R, F)
BATCH_SIZE = 32
VICINITY = 7
FEATURE_NUM = (VICINITY*2+1)**2
INPUT_SHAPE = (FEATURE_NUM*NUM_ACTIONS, ) # ((81*3), ))  (1, NUM_ACTIONS*FEATURE_NUM)
MEMORY_LENGTH = BATCH_SIZE*20

class MyPolicy(bp.Policy):
    """
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['batch_size'] = float(policy_args['batch_size']) if 'batch_size' in policy_args else BATCH_SIZE
        policy_args['vicinity'] = float(policy_args['vicinity']) if 'vicinity' in policy_args else VICINITY
        policy_args['learning_rate'] = float(policy_args['learning_rate']) if 'learning_rate' in policy_args else LEARNING_RATE
        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.Q = DQN.DQNetwork(input_shape=INPUT_SHAPE, alpha=0.5, gamma=0.8,
                               dropout_rate=0.1, num_actions=NUM_ACTIONS, batch_size=self.batch_size,
                               learning_rate=self.learning_rate)
        self.memory = []
        self.loss = []
        self.act2idx = {'L': 0, 'R': 1, 'F': 2}
        self.idx2act = {0: 'L', 1: 'R', 2: 'F'}
        self.memory_length = MEMORY_LENGTH

    def put_stats(self):  # TODO remove after testing
        pickle.dump(self.loss, open(self.dir_name + '/last_game_loss.pkl', 'wb'))
        pickle.dump(self.test(), open(self.dir_name + '/last_test_loss.pkl', 'wb'))
    #
    # def test(self):  # TODO REMOVE AFTER TESTING
    #     gt = self.Q.(self.replay_next, self.replay_reward, self.replay_idx)
    #     loss = self.model.train_on_batch(self.replay_prev[:self.replay_idx], gt)
    #     return loss


    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        if round >= self.batch_size:
            random_batches = np.random.choice(self.memory, self.batch_size)
            loss = self.Q.learn(random_batches)
            self.loss.append(loss)
        self.epsilon = self.epsilon * EPSILON_RATE

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

    def getFeatures(self, board, head):
        features = np.zeros([3, FEATURE_NUM])
        head_pos, direction = head
        for a in self.act2idx:
            moving_dir = bp.Policy.TURNS[direction][a]
            next_position = head_pos.move(moving_dir)
            map_after = self.getVicinityMap(board, next_position, moving_dir)  # map around head and turned so that snakes looks to the top
            features[self.act2idx[a]] = map_after.flatten()
        # print('f1: ', features.shape)
        features = features.flatten()
        # features = features/features.sum()
        # print('flatten: ', features.shape)
        return features


    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        board, head = new_state
        new_features = self.getFeatures(board, head)

        if round >=2:  # update to memory from previous round (prev_state)
            board_prev, head_prev = prev_state
            prev_features = self.getFeatures(board_prev, head_prev)
            memory_update = {'s_t': prev_features, 'a_t': prev_action, 'r_t': reward, 's_tp1': new_features}

            if len(self.memory) < self.memory_length:
                self.memory.append(memory_update)
            else:
                self.memory[round % self.memory_length] = memory_update

        if round == 4999:
            losses = self.loss
            with open(cwd + "/losses.pickle", "wb") as f:
                pickle.dump(losses, f)

        # act in new round, decide for new_state
        if np.random.rand() < self.epsilon:
            action = np.random.choice(bp.Policy.ACTIONS)

        else:
            # print(new_features.shape)
            q_values = self.Q.predict(new_features)
            # print(q_values)
            a_idx = np.argmax(q_values)
            # print(a_idx)
            # print('chose action: ' , self.idx2act[a_idx])
            action = self.idx2act[a_idx]

        return action
