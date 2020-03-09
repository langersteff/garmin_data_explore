from keras.layers import Conv1D, Conv2DTranspose, Dense, Flatten, Reshape, Layer, Conv2DTranspose, Lambda, Input
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import numpy as np

import keras.backend as K

from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)


def add_Conv1DTranspose(model, filters, kernel_size, strides=2, padding='same', activation='relu', name=None):
    model.add(Lambda(lambda x: K.expand_dims(x, axis=2)))
    model.add(Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding, activation=activation, name=name))
    model.add(Lambda(lambda x: K.squeeze(x, axis=2)))

def CAE(input_shape=(50, 9, 1), filters=[32, 64, 128, 10]):
    model = Sequential()

    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'

    print("input_shape", input_shape)

    model.add(Conv1D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))

    model.add(Conv1D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))

    model.add(Conv1D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))

    # DENSE
    model.add(Flatten())

    # EMBEDDING
    model.add(Dense(units=filters[3], name='embedding'))

    # DENSE
    model.add(Dense(units=filters[2]*int(input_shape[0]/8), activation='relu'))

    model.add(Reshape((int(input_shape[0]/8), filters[2])))
    add_Conv1DTranspose(model, filters=filters[1], kernel_size=3, strides=2, padding=pad3, activation='relu', name='deconv3')

    add_Conv1DTranspose(model, filters=filters[0], kernel_size=5, strides=2, padding='same', activation='relu', name='deconv2')

    add_Conv1DTranspose(model, filters=input_shape[1], kernel_size=5, strides=2, padding='same', name='deconv1')

    model.summary()
    return model

if __name__ == "__main__":
    from time import time

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default='usps', choices=['mnist', 'usps'])
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--save_dir', default='results/temp', type=str)
    args = parser.parse_args()
    print(args)

    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load dataset
    from datasets import load_mnist, load_usps
    if args.dataset == 'mnist':
        x, y = load_mnist()
    elif args.dataset == 'usps':
        x, y = load_usps('data/usps')

    # define the model
    model = CAE(input_shape=x.shape[1:], filters=[32, 64, 128, 10])
    plot_model(model, to_file=args.save_dir + '/%s-pretrain-model.png' % args.dataset, show_shapes=True)
    model.summary()

    # compile the model and callbacks
    optimizer = 'adam'
    model.compile(optimizer=optimizer, loss='mse')
    from keras.callbacks import CSVLogger
    csv_logger = CSVLogger(args.save_dir + '/%s-pretrain-log.csv' % args.dataset)

    # begin training
    t0 = time()
    model.fit(x, x, batch_size=args.batch_size, epochs=args.epochs, callbacks=[csv_logger])
    print('Training time: ', time() - t0)
    model.save(args.save_dir + '/%s-pretrain-model-%d.h5' % (args.dataset, args.epochs))

    # extract features
    feature_model = Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)
    features = feature_model.predict(x)
    print('feature shape=', features.shape)

    # use features for clustering
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=args.n_clusters)

    features = np.reshape(features, newshape=(features.shape[0], -1))
    pred = km.fit_predict(features)
    from . import metrics
    print('acc=', metrics.acc(y, pred), 'nmi=', metrics.nmi(y, pred), 'ari=', metrics.ari(y, pred))
