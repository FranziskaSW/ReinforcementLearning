from keras.models import Sequential
from keras.layers import *
import keras

class DQNetwork():
    def __init__(self, input_shape, alpha, gamma,
                 dropout_rate, num_actions, batch_size):
        self.alpha = alpha
        self.gamma = gamma
        self.dropout_rate = dropout_rate
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.batch_size = batch_size

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
            b = batch['s_tp1'][np.newaxis, ..., np.newaxis]
            y_t = batch['r_t'] + self.gamma * np.max(self.predict(b))
            y.append(y_t)

        x = np.array(x)[..., np.newaxis]
        y = np.array(y)

        self.model.fit(x, y, batch_size=self.batch_size, nb_epoch=1)

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
        s = state[np.newaxis, ..., np.newaxis]
        q_values = self.model.predict(s, batch_size=1)
        return q_values

