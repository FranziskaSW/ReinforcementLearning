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
        self.model.add(Dense(1, activation='softmax'))

        self.model.compile(loss='mean_squared_error',
                           optimizer='Adam')  # default learning rate = 0.01
        self.act2idx = {'L': 0, 'R': 1, 'F': 2}

    def learn(self, batches):
        x = []
        y = []
        for batch in batches:
            x.append(batch['s_t'])
            a_idx = self.act2idx[batch['a_t']]
            q_hat_vals = []
            #.... aber jetzt habe ich die vicinity map vom nächsten state nicht, weil ich nur die 9x9 karte gespeichert habe.. ok morgen weiter

            y_t = (batch['r_t'] + self.gamma * np.max(self.predict(batch['s_tp1']))) # here need to do loop as well, which one of the 3 actions gives best q-value.
            y.append(y_t)

        x = np.array(x)[..., np.newaxis]
        y = np.array(y)

        h = self.model.fit(x, y, batch_size=self.batch_size, epochs=1)
        return h.history['loss'][0]

    def predict(self, state):
        s = state[np.newaxis, ..., np.newaxis] # bring in right format
        q_values = self.model.predict(s, batch_size=1)
        return q_values

