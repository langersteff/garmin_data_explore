import numpy as np
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from .clusterer_base import ClustererBase
from .FIDCEC import FIDCEC
from keras.optimizers import SGD

class ClustererDCEC(ClustererBase):

    def fit_predict(self,
        data,
        data_prefix,
        num_clusters,
        latitudes,
        longitudes,
        y_true,
        optimizer = SGD(0.01, 0.9),
        loss="kld",
        filters=[32, 64, 128, 10],
        pretrain_optimizer = 'adam',
        update_interval = 140,
        pretrain_epochs = 300,
        batch_size = 256,
        tol = 0.001,
        maxiter = 2e4,
        save_dir = 'results'):

        print("\n---- %s ----" % self.title)

        dcec = FIDCEC(data[-1].shape,
                 filters=filters,
                 n_clusters=num_clusters,
                 alpha=1.0)

        dcec.pretrain(x=data, optimizer=pretrain_optimizer,
        epochs=pretrain_epochs, batch_size=batch_size,
        save_dir=save_dir)

        dcec.model.summary()
        dcec.compile(optimizer=optimizer, loss=loss)
        dcec.fit(data, y=None, tol=tol, maxiter=maxiter, batch_size=batch_size,
                        update_interval=update_interval, save_dir=save_dir)
        y_pred = dcec.y_pred

        np.save(self.filename_y_pred, y_pred)

        self.plot_y_pred(y_pred, latitudes, longitudes)
        self.save_scores(y_true, y_pred, data)

