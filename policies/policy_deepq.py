from keras.models import Sequential
from keras.layers import *
import keras
import numpy as np
from policies import base_policy as bp


EPSILON = 0.3
VICINITY = 5
DROPOUT_RATE = 0.2

class MyPolicy(bp.Policy):
    """
      A policy which avoids collisions with obstacles and other snakes. It has an epsilon parameter which controls the
      percentag of actions which are randomly chosen.
      """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['vicinity'] = float(policy_args['vicinity']) if 'vicinity' in policy_args else VICINITY
        policy_args['dropout_rate'] = float(policy_args['dropout_rate']) if 'dropout' in policy_args else DROPOUT_RATE
        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.input_shape = (self.vicinity*2+1, self.vicinity*2+1, 1)
        self.num_actions = len(['N', 'E', 'S', 'W'])
        self.id2dir = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}

    def dir2turn(self, head_dir, moving_dir):
        TURNS = {
            'N': {'N': 'F', 'E': 'R', 'W': 'L'},
            'E': {'E': 'F', 'S': 'R', 'N': 'L'},
            'S': {'S': 'F', 'W': 'R', 'E': 'L'},
            'W': {'W': 'F', 'N': 'R', 'S': 'L'}
        }
        turn = TURNS[head_dir][moving_dir]
        return turn

    def getVicinityMap(self):
        board = self.board
        vicinity = self.vicinity
        center = self.head

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

        return big_board[top:bottom, left:right]


    def cnn(self):
        """
        defines and trains a cnn model
        :param mnist_data: the MNIST data
        :param batch_size: batch size for training
        :param epochs: epochs of training
        :param dropout_rate: dropout rate between the layers
        :return: the trained cnn model and its training history
        """

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.dropout_rate))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.num_actions, activation='softmax'))
        
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='Adam',  # default learning rate = 0.01
                      metrics=['accuracy'])
        
        return model


    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

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
        head_pos, direction = head

        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        else:
            net = self.cnn() #  dont define new every time. save progress somewhere and update and whatever
            map = self.getVicinityMap(head_pos)
            map_resized = map[np.newaxis, ..., np.newaxis]  # (1, 11, 11, 1),
            # would like to do this: map.reshape(samples, input_shape[0], input_shape[1], input_shape[3) but doesnt work
            q_values = net.predict(map_resized)
            q_max, q_max_idx = np.max(q_values), np.argmax(q_values)
            moving_dir = self.id2dir[q_max_idx]
            action = self.dir2turn(head_dir=direction, moving_dir=moving_dir)

            return action

    def add_memory(self):
        """
        add action and so on to memory for memory replay
        :return:
        """
