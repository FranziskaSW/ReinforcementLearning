from keras.models import Sequential
from keras.layers import *
import keras
import numpy as np

def getVicinityMap(center, board):

    r, c = center
    left = c-vicinity
    right = c+vicinity+1
    top = r-vicinity
    bottom = r+vicinity+1


    big_board = board

    if left < 0:
        left_patch = np.matrix(big_board[:,left])
        left = 0
        right = 2*vicinity+1
        big_board = np.hstack([left_patch.T, big_board])

    if right >= board_size[1]:
        right_patch =  np.matrix(big_board[:, :(right%board_size[1]+1)])
        big_board = np.hstack([big_board, right_patch])

    if top < 0:
        top_patch =  np.matrix(big_board[top,:])
        top = 0
        bottom = 2*vicinity+1
        big_board = np.vstack([top_patch, big_board])

    if bottom >= board_size[0]:
        bottom_patch =  np.matrix(big_board[:(bottom%board_size[0])])
        big_board = np.vstack([big_board, bottom_patch])

    return big_board[top:bottom, left:right]

VicinityMap = getVicinityMap(curr_position, board)


def cnn(input_shape, dropout_rate):
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
                         input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='Adam',   # default learning rate = 0.01
                      metrics=['accuracy'])

    return model


img_rows, img_cols = vicinity*2+1, vicinity*2+1
input_shape = (img_rows, img_cols, 1)  # 1 will increase later with memory replay

net = cnn(input_shape=input_shape, dropout_rate=0.1)

samples = 1
map = getVicinityMap(curr_position, board)
# n = map.reshape(samples, img_rows, img_cols, 1) # somehow this does not work, but works for map = board[:28, :28].reshape(1, 28, 28, 1)
n = map[newaxis, ..., newaxis]
net.predict(n)