import numpy as np
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from .clusterer_base import ClustererBase
from CAEL2.CAEL2 import CAEL2
from keras.optimizers import SGD

class ClustererCAEL2(ClustererBase):

    def fit_predict(self,
        data,
        data_prefix,
        num_clusters,
        latitudes,
        longitudes,
        y_true,
        filters=[32, 64, 128, 10],
        batch_size = 32,
        epochs=500):

        print("\n---- %s ----" % self.title)
        print("Filters:", filters)
        print("Input Shape:", data[-1].shape)

        fit_cael2fi = isinstance(data, list)
        if fit_cael2fi:
            data, data_features = data
            filters.append(len(data_features[-1]))

        cael2 = CAEL2(data[-1].shape,
                 filters=filters,
                 n_clusters=num_clusters,
                 alpha=1.0)

        print(data.shape, "data.shape\n\n")
        #cael2.cae.summary()

        cael2.compile()
        cael2.fit(data, batch_size=batch_size, epochs=epochs)
        y_pred = cael2.predict(data)

        np.save(self.filename_y_pred, y_pred)

        self.plot_y_pred(y_pred, latitudes, longitudes)
        self.save_scores(y_true, y_pred, data)

