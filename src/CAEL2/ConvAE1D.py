from keras.layers import Conv1D, Conv1DTranspose, Dense, Flatten, Reshape, Layer, Input, Lambda
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import numpy as np
import keras.backend as K

def CAE(input_shape=(50, 9, 1), filters=[32, 64, 128, 10]):
    model = Sequential()

    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'

    model.add(Conv1D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))

    model.add(Conv1D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))

    model.add(Conv1D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))

    # DENSE
    model.add(Flatten())

    # EMBEDDING
    model.add(Dense(units=filters[3], name='raw_embedding'))

    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1), name='embedding'))

    # DENSE
    model.add(Dense(units=filters[2]*int(input_shape[0]/8), activation='relu'))

    model.add(Reshape((int(input_shape[0]/8), filters[2])))

    model.add(Conv1DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))

    model.add(Conv1DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))

    model.add(Conv1DTranspose(input_shape[1], 5, strides=2, padding='same', activation='relu', name='deconv1'))

    return model

