from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization, Activation, Flatten
from keras.optimizers import SGD
from sklearn.model_selection import KFold
from FIDCEC import FIDCEC
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, SpectralClustering, MeanShift
from sklearn.mixture import GaussianMixture
import pandas as pd
from geopy.distance import geodesic
from sklearn import metrics
import os


class MtbClassifier:

    def run_classification(self, X, y, classifiers, classifier_names, mtb_data_provider, mtb_visualizer=None, n_splits=15, window_length=50, step_size=.25, clear_outliers=False, print_plots=False, normalize_confusion_matrix='all'):

        # Create KFold Splits
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        kf.get_n_splits(X)

        # Run for every classifier
        for i in range(len(classifiers)):
            clf = classifiers[i]
            print("Classifier:", classifier_names[i])
            scores = []
            score_count = 0

            # Run for every split
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Windowing, Outlier Clearing
                X_train, y_train, _ = mtb_data_provider.create_training_data(X_train, y_train, window_length=window_length, clear_outliers = clear_outliers, step_size=step_size, calc_features=True)
                X_test, y_test, _ = mtb_data_provider.create_training_data(X_test, y_test, window_length=window_length, clear_outliers = clear_outliers, step_size=step_size, calc_features=True)

                # Oversample
                X_train, y_train = mtb_data_provider.evenly_oversample(X_train, y_train)
                X_test, y_test = mtb_data_provider.evenly_oversample(X_test, y_test)

                # Shuffle
                #X_train, y_train = shuffle(X_train, y_train)

                clf.fit(X_train, y_train)
                scores.append(clf.score(X_test, y_test))
                score_count += 1
                print("Score:", clf.score(X_test, y_test))

                if print_plots:
                    unique, counts = np.unique(y_train, return_counts=True)
                    y_test_pred = clf.predict(X_test)
                    mtb_visualizer.print_confusion_matrix(y_test, y_test_pred, unique.tolist())

            score = np.sum(scores) / score_count
            print("Avg Score:", score)
            print("Min", np.min(scores))
            print("Max", np.max(scores))
            print("Median", np.median(scores))
            print('------------------------------------------------\n\n')


    def create_cnn_model(self, input_shape, num_classes, n_conv_blocks=2):
        max_features = 5000
        maxlen = 400
        batch_size = 32
        embedding_dims = 50
        filters = input_shape[1]
        kernel_size = 8
        hidden_dims = 128
        epochs = 2
        dropout = .2

        model = Sequential()

        model.add(Conv1D(filters,
                        kernel_size,
                        input_shape=input_shape,
                        padding='same',
                        strides=1))

        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout))

        for n in range(1, n_conv_blocks):
            model.add(Conv1D(filters**n,
                        kernel_size,
                        padding='same',
                        strides=1))

            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(dropout))

        model.add(Flatten())
        model.add(Dense(hidden_dims))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

        model.add(Dense(num_classes, activation='softmax'))

        print(model.summary())

        opt = SGD(lr=0.01)
        model.compile(loss = "categorical_crossentropy", optimizer = opt)

        model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

        return model

    def run_cnn_model(self, X, y, mtb_data_provider, mtb_visualizer, n_splits=2, window_length=50, step_size=.25, clear_outliers=False, n_conv_blocks=2):
        #print(X.shape)

        # Create KFold Splits
        kf = KFold(n_splits=n_splits, shuffle=False)
        kf.get_n_splits(X)

        #! In some splits it could be, that there is a label missing
        unique, _ = np.unique(y, return_counts=True)
        model = self.create_cnn_model((window_length, X.shape[1]), np.max(unique) + 1, n_conv_blocks=n_conv_blocks)


        # Run for every split
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Windowing, Outlier Clearing
            X_train, y_train, _ = mtb_data_provider.create_training_data(X_train, y_train, window_length=window_length, clear_outliers = clear_outliers, step_size=step_size)
            X_test, y_test, _ = mtb_data_provider.create_training_data(X_test, y_test, window_length=window_length, clear_outliers = clear_outliers, step_size=step_size)

            # Oversample
            X_train, y_train = mtb_data_provider.evenly_oversample(X_train, y_train)
            X_test, y_test = mtb_data_provider.evenly_oversample(X_test, y_test)

            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)

            history = model.fit(X_train, y_train, batch_size = 32, epochs = 1000, validation_data=(X_test, y_test), verbose=0)

            # Plot validation acc and loss
            plt.plot(history.history['val_accuracy'])
            plt.plot(history.history['val_loss'])
            plt.legend(['acc', 'loss'])
            plt.show()

    ##############################################
    ######## UNSUPERVISED DEEP CLUSTERING ########
    ##############################################

    def train_and_compare_fidcec(self,
                                mtb_data_provider,
                                prefix = '',
                                run_fidcec=True,
                                run_dcec=True,
                                run_classical=True,
                                window_lengths=[100,200],
                                sub_sample_lengths=[50,100],
                                num_clusters=3,
                                verbose = True,
                                update_interval=140,
                                save_dir='results/temp',
                                tol=0.001,
                                maxiter=2e4,
                                optimizer='adam',
                                loss=['kld', 'mse'],
                                loss_weights=[.1, 1]):

        for window_length in window_lengths:
            for sub_sample_length in sub_sample_lengths:
                if sub_sample_length >= window_length:
                    continue

                print("----------------------------------------------------------------")
                print("window_length:", window_length)
                print("sub_sample_length:", sub_sample_length)
                print("clusters:", num_clusters)

                data_prefix = "%s_%s_%s" % (prefix, str(window_length), str(sub_sample_length))
                experiment_prefix = "%s_%s" % (data_prefix, str(num_clusters))

                filename_raw = "data/%s_raw.npy" % data_prefix
                filename_features = "data/%s_features.npy" % data_prefix
                filename_labels = "data/%s_labels.npy" % data_prefix

                # Load pre saved samples from npy files
                data_raw = np.load(filename_raw)
                feature_file = np.load(filename_features)
                data_features = feature_file[:, :-2]
                null_features = np.zeros(data_features.shape)
                y_true = np.load(filename_labels)

                if run_fidcec:
                    print("\n---- FIDCEC ----")
                    filename_y_pred_fidcec = "evaluation/%s_fidcec_y_pred" % experiment_prefix

                    if os.path.isfile(filename_y_pred_fidcec + '.npy'):
                        y_pred = np.load(filename_y_pred_fidcec + '.npy')
                    else:
                        fidcec = FIDCEC([data_raw[0].shape, data_features[0].shape], n_clusters=num_clusters)
                        fidcec.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)
                        fidcec.fit([data_raw, data_features], y=None,
                                tol=tol,
                                maxiter=maxiter,
                                update_interval=update_interval,
                                save_dir=save_dir,
                                verbose = verbose,
                                cae_weights=None)
                        y_pred = fidcec.y_pred
                        np.save(filename_y_pred_fidcec, y_pred)

                    fig1 = figure(1, figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
                    fig1.suptitle('FIDCEC', fontsize=20)
                    geometry = gpd.points_from_xy(feature_file[:, -1], feature_file[:, -2])
                    gdf = GeoDataFrame(geometry=geometry)
                    gdf.plot(c=y_pred, figsize=(20, 30))

                    adjusted_rand_score = metrics.adjusted_rand_score(y_true, y_pred)
                    print("Adjusted rand score: ", adjusted_rand_score)
                    np.save(filename_y_pred_fidcec + str(adjusted_rand_score), adjusted_rand_score)

                if run_dcec:
                    print("\n---- DCEC ----")
                    filename_y_pred_dcec = "evaluation/%s_dcec_y_pred" % experiment_prefix

                    if os.path.isfile(filename_y_pred_dcec + '.npy'):
                        y_pred = np.load(filename_y_pred_dcec + '.npy')
                    else:
                        fidcec = FIDCEC([data_raw[0].shape, null_features[0].shape], n_clusters=num_clusters)
                        fidcec.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)
                        fidcec.fit([data_raw, null_features], y=None,
                                tol=tol,
                                maxiter=maxiter,
                                update_interval=update_interval,
                                save_dir=save_dir,
                                verbose = verbose,
                                cae_weights=None)
                        y_pred = fidcec.y_pred
                        np.save(filename_y_pred_dcec, filename_y_pred_dcec)

                    fig2 = figure(2, figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
                    fig1.suptitle('DCEC', fontsize=20)
                    geometry = gpd.points_from_xy(feature_file[:, -1], feature_file[:, -2])
                    gdf = GeoDataFrame(geometry=geometry)
                    gdf.plot(c=y_pred, figsize=(20, 30))

                    adjusted_rand_score = metrics.adjusted_rand_score(y_true, y_pred)
                    print("Adjusted rand score: ", adjusted_rand_score)
                    np.save(filename_y_pred_dcec + str(adjusted_rand_score), adjusted_rand_score)

                if run_classical:
                    print("\n---- GaussianMixture on Features ----")
                    filename_y_pred_classical = "evaluation/%s_gaussian_y_pred" % experiment_prefix

                    if os.path.isfile(filename_y_pred_classical + '.npy'):
                        y_pred = np.load(filename_y_pred_classical + '.npy')
                    else:
                        clusterer = GaussianMixture(n_components=num_clusters)
                        y_pred_classical = clusterer.fit_predict(data_features)
                        np.save(filename_y_pred_classical, y_pred_classical)

                    fig3 = figure(3, figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
                    fig1.suptitle('GaussianMixture', fontsize=20)
                    geometry = gpd.points_from_xy(feature_file[:, -1], feature_file[:, -2])
                    gdf = GeoDataFrame(geometry=geometry)
                    gdf.plot(c=y_pred_classical, figsize=(20, 30))

                    adjusted_rand_score = metrics.adjusted_rand_score(y_true, y_pred)
                    print("Adjusted rand score: ", adjusted_rand_score)
                    np.save(filename_y_pred_classical + str(adjusted_rand_score), adjusted_rand_score)

                filename_fidced = "evaluation/%s_fidcec_compare.png" % experiment_prefix
                plt.savefig(filename_fidced)
                plt.show()
