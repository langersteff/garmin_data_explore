import numpy as np
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from .clusterer_base import ClustererBase
from DEC.DEC import DEC
from keras.optimizers import SGD

class ClustererDEC(ClustererBase):

    def fit_predict(self,
        data,
        data_prefix,
        num_clusters,
        latitudes,
        longitudes,
        y_true,
        optimizer = SGD(0.01, 0.9),
        loss="kld",
        dec_dims=[500, 500, 2000, 10],
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

        print("\n---- %s ----" % self.title)

        self.dec_dims = np.hstack((data.shape[-1], dec_dims))

        dec = DEC(dims=self.dec_dims, n_clusters=num_clusters, init=init)
        dec.pretrain(x=data, y=None, optimizer=pretrain_optimizer,
        epochs=pretrain_epochs, batch_size=batch_size,
        save_dir=save_dir)

        dec.model.summary()
        dec.compile(optimizer=optimizer, loss=loss)
        y_pred = dec.fit(data, y=None, tol=tol, maxiter=maxiter, batch_size=batch_size,
                        update_interval=update_interval, save_dir=save_dir)
        np.save(self.filename_y_pred, y_pred)

        self.plot_y_pred(y_pred, latitudes, longitudes)
        self.save_scores(y_true, y_pred, data)

