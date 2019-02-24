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
INPUT_SHAPE = (20, 20, 1)
NUM_ACTIONS = 3
BATCH_SIZE = 32

class MyPolicy(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes. It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.Q = DQN.DQNetwork(input_shape=(20, 20, 1), alpha=0.5, gamma=0.5,
                               dropout_rate=0.1, num_actions=3, batch_size=32)
        # self.alpha = ALPHA
        # self.gamma = GAMMA
        # self.dropout_rate = DROPOUT_RATE
        # self.input_shape = INPUT_SHAPE
        # self.num_actions = NUM_ACTIONS
        # self.batch_size = BATCH_SIZE

        # self.model = Sequential()
        # self.model.add(Conv2D(32, kernel_size=(3, 3),
        #                       activation='relu',
        #                       input_shape=self.input_shape))
        # self.model.add(Conv2D(64, (3, 3), activation='relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Dropout(self.dropout_rate))
        # self.model.add(Flatten())
        # self.model.add(Dense(128, activation='relu'))
        # self.model.add(Dropout(self.dropout_rate))
        # self.model.add(Dense(self.num_actions, activation='softmax'))
        #
        # self.model.compile(loss=keras.losses.categorical_crossentropy,
        #                    optimizer='Adam',  # default learning rate = 0.01
        #                    metrics=['accuracy'])

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
            q_values = self.Q.predict(board)
            idx = np.argmax(q_values)
            return bp.Policy.ACTIONS[idx]