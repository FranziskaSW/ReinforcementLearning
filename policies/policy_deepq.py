from keras.models import Sequential
from keras.layers import *
import keras
import numpy as np
from policies import base_policy as bp

EPSILON = 0.3
VICINITY = 5
DROPOUT_RATE = 0.2

class DQNetwork():
    def __init__(self, input_shape, alpha=0.1, gamma=0.99,
                 dropout_prob=0.1):

        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(Dense(self.num_actions, activation='softmax'))

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='Adam',  # default learning rate = 0.01
                      metrics=['accuracy'])

    def learn(self, batches):
        x = []
        y = []
        
        for batch in batches:  # ((s_t, a_t) = q_t, r_t, s_(t+1))
            # calculate the gradiend step: y_t = r_t + gamma * Q_(t+1)
            x.append(batch['s_t'])
            q_tp1 = np.max(self.predict(batch['s_tp1']))

        # for datapoint in batch:
        #
        #     # The error must be 0 on all actions except the one taken
        #     t = list(self.predict(datapoint['source'])[0])
        #     if datapoint['final']:
        #         t[datapoint['action']] = datapoint['reward']
        #     else:
        #         t[datapoint['action']] = datapoint['reward'] + \
        #                                  self.gamma * next_q_value
        #
        #     t_train.append(t)
        #
        #     # Prepare inputs and targets
        # x_train = np.asarray(x_train).squeeze()
        # t_train = np.asarray(t_train).squeeze()
        #
        # # Train the model for one epoch
        # h = self.model.fit(x_train,
        #                    t_train,
        #                    batch_size=32,
        #                    nb_epoch=1)

    def predict(self, state):
        q_values = self.model.predict(state, batch_size=1)
        return q_values




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
        self.memory = []
        self.network = DQNetwork(self.input_shape, self.dropout_rate)
        print(self.network.summary())

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



    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        random_batches = np.random.choice(self.memory, self.batch_size)

        self.network.learn(random_batches)

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

    def getQValue(self, state):
        q_values = self.model.predict(state, batch_size=1)
        q_max, q_max_idx = np.max(q_values), np.argmax(q_values)
        moving_dir = self.id2dir[q_max_idx]
        return q_value, moving_dir

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        board, head = new_state
        head_pos, direction = head
        state_before = self.getVicinityMap(head_pos)

        if np.random.rand() < self.epsilon:
            action = np.random.choice(bp.Policy.ACTIONS)
            moving_dir = bp.Policy.TURNS[direction][action]

        else:
            q_value, moving_dir = getQValue(state_before)
            action = self.dir2turn(head_dir=direction, moving_dir=moving_dir)

        next_position = bp.Policy.Position.move(moving_dir)
        state_after = self.getVicinityMap(next_position)

        self.memory += [{'s_t' : state_before, 'a_t' : action, 'r_t' : reward, 's_tp1' : state_after}]
        return action

    def add_memory(self, ):
        """
        add transition to memory for memory replay
        :return:

        """
