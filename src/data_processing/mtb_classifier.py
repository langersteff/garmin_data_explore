from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization, Activation, Flatten
from keras.layers import Concatenate, Input, concatenate
from keras.optimizers import SGD
from sklearn.model_selection import KFold
from DEC.DEC import DEC
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, SpectralClustering, MeanShift, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import pandas as pd
from geopy.distance import geodesic
from sklearn import metrics
import os
from time import time


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


    def create_cnn_model(self, input_shape, num_classes, features_shape=None, n_conv_blocks=2):
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

        softmax_layer = Dense(num_classes, activation='softmax')

        final_model = None

        if features_shape is not None:
            # Get Features
            handcrafted_features = Input(shape=features_shape, name='handcrafted_features')

            # Concat hidden and feature_layer
            concat_layer = concatenate([model.output, handcrafted_features])
            model.add(concat_layer)

            self.final_model = Model(inputs=[model.input, handcrafted_features],
                            outputs=[softmax_layer])
        else:
            final_model = model


        print(final_model.summary())

        opt = SGD(lr=0.01)
        final_model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

        return final_model

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

    def train_and_compare_unsupervised_clusterings(self,
                                mtb_data_provider,
                                force_overwrite = False,
                                prefix = '',
                                run_dec=True,
                                run_fidec=True,
                                run_classical=True,
                                window_lengths=[100,200],
                                sub_sample_lengths=[50,100],
                                nums_clusters=[3],
                                verbose = True,
                                update_interval=140,
                                save_dir='results/temp',
                                tol=0.001,
                                maxiter=2e4,
                                optimizer='adam',
                                loss=['kld', 'mse'],
                                loss_weights=[.1, 1]):

        for num_clusters in nums_clusters:
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
                    print("y_true", np.unique(y_true, return_counts=True))

                    fig_truth = figure(4, figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
                    fig_truth.suptitle('Ground Truth', fontsize=20)
                    geometry = gpd.points_from_xy(feature_file[:, -1], feature_file[:, -2])
                    gdf = GeoDataFrame(geometry=geometry)
                    gdf.plot(c=y_true, figsize=(20, 30))

                    init = 'glorot_uniform'
                    pretrain_optimizer = 'adam'
                    update_interval = 140
                    pretrain_epochs = 300
                    batch_size = 256
                    tol = 0.001
                    maxiter = 2e4
                    save_dir = 'results'

                    if run_dec:
                        print("\n---- DEC ----")
                        filename_y_pred_dec = "evaluation/%s_dec_y_pred" % experiment_prefix

                        if os.path.isfile(filename_y_pred_dec + '.npy') and not force_overwrite:
                            y_pred = np.load(filename_y_pred_dec + '.npy')
                        else:
                            data_raw_flat = data_raw.reshape((data_raw.shape[0] * data_raw.shape[1], data_raw.shape[2]))
                            dec = DEC(dims=[data_raw_flat.shape[-1], 500, 500, 2000, 10], n_clusters=num_clusters, init=init)
                            dec.pretrain(x=data_raw_flat, y=None, optimizer=pretrain_optimizer,
                            epochs=pretrain_epochs, batch_size=batch_size,
                            save_dir=save_dir)

                            dec.model.summary()
                            t0 = time()
                            dec.compile(optimizer=SGD(0.01, 0.9), loss='kld')
                            y_pred = dec.fit(data_raw_flat, y=None, tol=tol, maxiter=maxiter, batch_size=batch_size,
                                            update_interval=update_interval, save_dir=save_dir)
                            print("y_pred", np.unique(y_pred, return_counts=True))
                            np.save(filename_y_pred_dec, y_pred)

                        fig1 = figure(1, figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
                        fig1.suptitle('DEC', fontsize=20)
                        gdf = GeoDataFrame(geometry=geometry)
                        gdf.plot(c=y_pred, figsize=(20, 30))

                        self.save_scores(filename_y_pred_dec + "_score", y_true, y_pred, data_features)

                    if run_fidec:
                        print("\n---- FIDEC ----")
                        filename_y_pred_fidec = "evaluation/%s_fidec_y_pred" % experiment_prefix

                        if os.path.isfile(filename_y_pred_fidec + '.npy') and not force_overwrite:
                            y_pred = np.load(filename_y_pred_fidec + '.npy')
                        else:
                            data_raw_flat = data_raw.reshape((data_raw.shape[0] * data_raw.shape[1], data_raw.shape[2]))
                            fidec = DEC(dims=[data_raw_flat.shape[-1], 500, 500, 2000, 10], feature_dims=data_features[0].shape, n_clusters=num_clusters, init=init)
                            fidec.pretrain(x=[data_raw_flat, data_features], y=None, optimizer=pretrain_optimizer,
                            epochs=pretrain_epochs, batch_size=batch_size,
                            save_dir=save_dir)

                            fidec.model.summary()
                            t0 = time()
                            fidec.compile(optimizer=SGD(0.01, 0.9), loss='kld')
                            y_pred = fidec.fit(data_raw_flat, y=None, tol=tol, maxiter=maxiter, batch_size=batch_size,
                                            update_interval=update_interval, save_dir=save_dir)
                            print("y_pred", np.unique(y_pred, return_counts=True))
                            np.save(filename_y_pred_fidec, y_pred)

                        fig1 = figure(1, figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
                        fig1.suptitle('FIDEC', fontsize=20)
                        gdf = GeoDataFrame(geometry=geometry)
                        gdf.plot(c=y_pred, figsize=(20, 30))

                        self.save_scores(filename_y_pred_fidec + "_score", y_true, y_pred, data_features)



                    if run_classical:
                        print("\n---- Classical SciKit Clustering on Features ----")
                        filename_y_pred_classical = "evaluation/%s_classical_y_pred" % experiment_prefix

                        if os.path.isfile(filename_y_pred_classical + '.npy') and not force_overwrite:
                            y_pred = np.load(filename_y_pred_classical + '.npy')
                        else:
                            clusterer = AgglomerativeClustering(n_clusters=num_clusters)
                            y_pred = clusterer.fit_predict(data_features)
                            np.save(filename_y_pred_classical, y_pred)

                        fig3 = figure(3, figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
                        fig1.suptitle('AgglomerativeClustering', fontsize=20)
                        gdf = GeoDataFrame(geometry=geometry)
                        gdf.plot(c=y_pred, figsize=(20, 30))

                        self.save_scores(filename_y_pred_classical + "_score", y_true, y_pred, data_features)

                    # TODO: This doesn't properly save plots somehow
                    filename_fidec = "evaluation/%s_fidec_compare.png" % experiment_prefix
                    plt.savefig(filename_fidec)
                    plt.show()

    def save_scores(self, filename, y_true, y_pred, data):
        scores = []
        scores.append(['score_name', 'score_result'])
        scores.append(['adjusted_rand_score', metrics.adjusted_rand_score(y_true, y_pred)])
        scores.append(['adjusted_mutual_info_score', metrics.adjusted_mutual_info_score(y_true, y_pred)])
        scores.append(['homogeneity_score', metrics.homogeneity_score(y_true, y_pred)])
        scores.append(['completeness_score', metrics.completeness_score(y_true, y_pred)])
        scores.append(['v_measure_score', metrics.v_measure_score(y_true, y_pred)])
        scores.append(['fowlkes_mallows_score', metrics.fowlkes_mallows_score(y_true, y_pred)])
        scores.append(['silhouette_score', metrics.silhouette_score(data, y_pred)])
        scores.append(['davies_bouldin_score', metrics.davies_bouldin_score(data, y_pred)])
        scores.append(['calinski_harabasz_score', metrics.calinski_harabasz_score(data, y_pred)])

        print(scores)

        np.savetxt(filename + ".csv", scores, fmt='%s,%s', delimiter=",")