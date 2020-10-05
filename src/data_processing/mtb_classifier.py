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
from .clusterer_FIDCEC import ClustererFIDCEC
from .clusterer_DCEC import ClustererDCEC
from .clusterer_DEC import ClustererDEC
from .clusterer_FIDEC import ClustererFIDEC
from .clusterer_Classical import ClustererClassical
from .clusterer_CAEL2 import ClustererCAEL2
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
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class MtbClassifier:

    def run_classification(self,
        classifiers,
        classifier_names,
        mtb_data_provider,
        dataset_input='',
        input_columns=[],
        label_column='osm_mtb:scale',
        ignore_label=0,
        window_lengths=[100,200],
        sub_sample_lengths=[50,100],
        mtb_visualizer=None,
        n_splits=15,
        step_size=.25,
        clear_outliers=False,
        print_plots=False,
        normalize_confusion_matrix='all'):

        print(label_column)
        df = pd.read_csv('data/' + dataset_input + '.csv')
        X = df[input_columns].values
        y = df[label_column].values
        y_mask = y != 0
        y = y[y_mask]
        print(np.unique(y, return_counts=True))
        y = LabelEncoder().fit_transform(y)
        X = X[y_mask]

        # Normalize, despite the first and the last three fields (Timestamp, Heading in rad, Latitude, Longitude)
        raw_scaler = StandardScaler()
        X_norm = raw_scaler.fit_transform(X[:, 1:-3])

        print("Fitting pca...", X_norm.shape)
        pca = PCA(n_components=X_norm.shape[1], random_state=42)
        X_norm = pca.fit_transform(X_norm)

        # Run for every classifier
        for i in range(len(classifiers)):
            clf = classifiers[i]
            print("Classifier:", classifier_names[i])
            scores = []
            score_count = 0

            for window_length in window_lengths:
                for sub_sample_length in sub_sample_lengths:
                    if sub_sample_length > window_length:
                        continue

                    print("----------------------------------------------------------------")
                    print("window_length:", window_length)
                    print("sub_sample_length:", sub_sample_length)

                    # Create KFold Splits
                    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                    kf.get_n_splits(X_norm)

                    # Run for every split
                    for train_index, test_index in kf.split(X_norm):
                        X_train, X_test = X_norm[train_index], X_norm[test_index]
                        y_train, y_test = y[train_index], y[test_index]

                        #X_train = pca.transform(X_train)
                        #X_test = pca.transform(X_test)

                        X_train, y_train, _, _ = mtb_data_provider.create_training_data(X_train, y_train, window_length=window_length, clear_outliers = clear_outliers, step_size=step_size, calc_features=False)
                        X_test, y_test, _, _ = mtb_data_provider.create_training_data(X_test, y_test, window_length=window_length, clear_outliers = clear_outliers, step_size=step_size, calc_features=False)

                        y_train = np.ndarray.flatten(y_train)
                        y_test = np.ndarray.flatten(y_test)

                        # Oversample
                        X_train, y_train = mtb_data_provider.evenly_oversample(X_train, y_train)
                        X_test, y_test = mtb_data_provider.evenly_oversample(X_test, y_test)

                        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
                        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

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
                                dataset_input = '',
                                label_column = 0,
                                ignore_label='0',
                                run_dec=False,
                                run_fidec=False,
                                run_dcec=False,
                                run_fidcec=False,
                                run_drec=False,
                                run_fidrec=False,
                                run_classical_raw=False,
                                run_classical_raw_fi=False,
                                run_classical_features=False,
                                run_cael2=False,
                                run_cael2fi=False,
                                dec_dims=[500, 500, 2000, 10],
                                dcec_filters=[32, 64, 128, 10],
                                cael2_filters=[32, 64, 128, 10],
                                window_lengths=[100,200],
                                sub_sample_lengths=[50,100],
                                init = 'glorot_uniform',
                                pretrain_optimizer = 'adam',
                                update_interval = 140,
                                pretrain_epochs = 300,
                                batch_size = 256,
                                tol = 0.001,
                                maxiter = 2e4,
                                save_dir = 'results'):

        # TODO: Instead of running some random number of clusters, read the amount of clusters from the label file and cluster to this amount
        for window_length in window_lengths:
            for sub_sample_length in sub_sample_lengths:
                if sub_sample_length > window_length:
                    continue

                print("----------------------------------------------------------------")
                print("window_length:", window_length)
                print("sub_sample_length:", sub_sample_length)


                data_prefix = "%s_%s_%s" % (dataset_input, str(window_length), str(sub_sample_length))

                filename_raw = "data/%s_raw.npy" % data_prefix
                filename_features = "data/%s_features.npy" % data_prefix
                filename_labels = "data/%s_labels.npy" % data_prefix

                y_true = np.load(filename_labels)[:, label_column]
                y_true_mask = y_true != ignore_label
                y_true = y_true[y_true_mask]
                print("clusters:", np.unique(y_true, return_counts=True))

                y_true = LabelEncoder().fit_transform(y_true)
                num_clusters = len(np.unique(y_true, return_counts=True)[0])

                # Load pre saved samples from npy files
                data_raw = np.load(filename_raw)[:, :, :-3]
                data_raw = data_raw[y_true_mask]
                data_raw_flat = data_raw.reshape((data_raw.shape[0], data_raw.shape[1] * data_raw.shape[2]))
                feature_file = np.load(filename_features)
                data_features = feature_file[:, :-2] # The last two values are latitude, longitude
                data_features = data_features[y_true_mask]

                # Oversample?
                # data_raw, y_true = mtb_data_provider.evenly_oversample(data_raw, y_true)
                # data_features, y_true = mtb_data_provider.evenly_oversample(data_features, y_true)

                print("Raw data shape: ", data_raw.shape)
                print("Features shape: ", data_features.shape)


                #ClustererClassical().plot_y_pred(y_true, feature_file[:, -1], feature_file[:, -2], "Ground truth")

                print_summary, print_unique_pred, plot_pred, save_pred = False, True, False, True

                if run_dec:
                    dec = ClustererDEC(print_summary=print_summary, print_unique_pred=print_unique_pred, plot_pred=plot_pred, save_pred=save_pred)
                    dec.set_prefixes(dataset_input, num_clusters, window_length, sub_sample_length, 'dec')
                    dec.fit_predict(data_raw_flat, data_prefix, num_clusters, feature_file[:, -1], feature_file[:, -2], y_true, dec_dims=dec_dims)

                if run_fidec:
                    fidec = ClustererFIDEC(print_summary=print_summary, print_unique_pred=print_unique_pred, plot_pred=plot_pred, save_pred=save_pred)
                    fidec.set_prefixes(dataset_input, num_clusters, window_length, sub_sample_length, 'fidec')
                    fidec.fit_predict([data_raw_flat, data_features], data_prefix, num_clusters, feature_file[:, -1], feature_file[:, -2], y_true, dec_dims=dec_dims)

                if run_dcec:
                    dec = ClustererDCEC(print_summary=print_summary, print_unique_pred=print_unique_pred, plot_pred=plot_pred, save_pred=save_pred)
                    dec.set_prefixes(dataset_input, num_clusters, window_length, sub_sample_length, 'dcec')
                    dec.fit_predict(data_raw, data_prefix, num_clusters, feature_file[:, -1], feature_file[:, -2], y_true, filters=np.hstack((dcec_filters, data_features.shape[-1])))

                if run_fidcec:
                    fidec = ClustererFIDCEC(print_summary=print_summary, print_unique_pred=print_unique_pred, plot_pred=plot_pred, save_pred=save_pred)
                    fidec.set_prefixes(dataset_input, num_clusters, window_length, sub_sample_length, 'fidcec')
                    fidec.fit_predict([data_raw, data_features], data_prefix, num_clusters, feature_file[:, -1], feature_file[:, -2], y_true, filters=dcec_filters)

                if run_classical_raw:
                    classical_raw = ClustererClassical(print_summary=print_summary, print_unique_pred=print_unique_pred, plot_pred=plot_pred, save_pred=save_pred)
                    classical_raw.set_prefixes(dataset_input, num_clusters, window_length, sub_sample_length, 'classical_raw')
                    classical_raw.fit_predict(data_raw_flat, data_prefix, num_clusters, feature_file[:, -1], feature_file[:, -2], y_true, pca_n_components=data_features.shape[-1])

                if run_classical_raw_fi:
                    classical_fi = ClustererClassical(print_summary=print_summary, print_unique_pred=print_unique_pred, plot_pred=plot_pred, save_pred=save_pred)
                    classical_fi.set_prefixes(dataset_input, num_clusters, window_length, sub_sample_length, 'classical_raw_fi')
                    classical_fi.fit_predict([data_raw_flat, data_features], data_prefix, num_clusters, feature_file[:, -1], feature_file[:, -2], y_true, pca_n_components=data_features.shape[-1])

                if run_classical_features:
                    classical_fi = ClustererClassical(print_summary=print_summary, print_unique_pred=print_unique_pred, plot_pred=plot_pred, save_pred=save_pred)
                    classical_fi.set_prefixes(dataset_input, num_clusters, window_length, sub_sample_length, 'classical_features')
                    classical_fi.fit_predict(data_features, data_prefix, num_clusters, feature_file[:, -1], feature_file[:, -2], y_true)

                if run_cael2:
                    cael2_raw = ClustererCAEL2(print_summary=print_summary, print_unique_pred=print_unique_pred, plot_pred=plot_pred, save_pred=save_pred)
                    cael2_raw.set_prefixes(dataset_input, num_clusters, window_length, sub_sample_length, 'cael2')
                    cael2_raw.fit_predict(data_raw, data_prefix, num_clusters, feature_file[:, -1], feature_file[:, -2], y_true, filters=np.hstack((cael2_filters, data_features.shape[-1])))

                if run_cael2fi:
                    cael2_raw_fi = ClustererCAEL2(print_summary=print_summary, print_unique_pred=print_unique_pred, plot_pred=plot_pred, save_pred=save_pred)
                    cael2_raw_fi.set_prefixes(dataset_input, num_clusters, window_length, sub_sample_length, 'cael2fi')
                    cael2_raw_fi.fit_predict([data_raw, data_features], data_prefix, num_clusters, feature_file[:, -1], feature_file[:, -2], y_true, filters=cael2_filters)
