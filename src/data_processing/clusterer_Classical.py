import numpy as np
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from .clusterer_base import ClustererBase
from keras.optimizers import SGD
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA

from numpy.random import seed
seed(42)
np.random.seed(42)

class ClustererClassical(ClustererBase):

    def fit_predict(self,
        data,
        data_prefix,
        num_clusters,
        latitudes,
        longitudes,
        y_true,
        pca_n_components=None):

        print("\n---- %s ----" % self.title)

        # Check if Feature Injection should be aplied
        is_classical_raw_fi = len(data) == 2

        # Check if Raw and Features data was pased
        if is_classical_raw_fi:
            data, data_features = data

        # Check if a PCA should be applied (This is not the case for Features-Only)
        if pca_n_components is not None:
            print("PCA on components:", pca_n_components)
            pca = PCA(n_components=pca_n_components, random_state=42)
            data = pca.fit_transform(data)

        # Stack PCA and handcrafted features
        if is_classical_raw_fi:
            data = np.hstack((data, data_features))

        clusterer = KMeans(n_clusters=num_clusters)
        y_pred = clusterer.fit_predict(data)
        np.save(self.filename_y_pred, y_pred)

        self.plot_y_pred(y_pred, latitudes, longitudes)
        self.save_scores(y_true, y_pred, data)
        # TODO: Which data to use for calculating the scores?

