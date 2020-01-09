from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import numpy as np

class MtbClassifier:

    def run_classification(self, X, y, classifiers, classifier_names, mtb_data_provider, n_splits=15, window_length=100, step_size=.25, clear_outliers=False, print_plots=False):
        
        # Create KFold Splits
        kf = KFold(n_splits=n_splits, shuffle=False, random_state=42) 
        kf.get_n_splits(X)

        # Run for every classifier
        for i in range(len(classifiers)):
            clf = classifiers[i]
            print("Classifier:", classifier_names[i])
            scores = 0.0
            score_count = 0

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

                # Shuffle
                X_train, y_train = shuffle(X_train, y_train)

                unique, counts = np.unique(y_train, return_counts=True)
                print(unique, counts)

                clf.fit(X_train, y_train)
                print("Score:", clf.score(X_test, y_test))
                scores += clf.score(X_test, y_test)
                score_count += 1

                if print_plots:
                    y_test_pred = clf.predict(X_test)
                    mtb_visualizer.print_confusion_matrix(y_test, y_test_pred, [0,1])

            score = scores / score_count
            print("Avg Score:", score)
            print('------------------------------------------------\n\n')
            
