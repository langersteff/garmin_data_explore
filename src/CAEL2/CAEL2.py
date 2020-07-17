from time import time
import numpy as np
from keras.models import Model
from sklearn.cluster import KMeans
from .ConvAE1D import CAE

class CAEL2(object):
    def __init__(self,
                 input_shape,
                 filters=[32, 64, 128, 10],
                 n_clusters=10,
                 alpha=1.0):

        super().__init__()

        self.n_clusters = n_clusters
        self.input_shape = input_shape

        self.cae = CAE(input_shape, filters)
        self.embedding = self.cae.get_layer(name='embedding').output
        self.encoder = Model(inputs=self.cae.input, outputs=self.embedding)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    def compile(self, loss='mse', optimizer='adam'):
        self.cae.compile(optimizer=optimizer, loss=loss)

    def fit(self, inputs, batch_size=32, epochs=500):

        fit_cael2fi = isinstance(inputs, list)
        if not fit_cael2fi: #CAEL2
            x = inputs
        else: #CAEL2FI
            x = inputs[0]
            features = inputs[1]

        self.cae.fit(x, x, batch_size=batch_size, epochs=epochs)
        embedded_features = self.encoder.predict(x)
        self.kmeans.fit(embedded_features)

    def predict(self, x):
        embedded_features = self.encoder.predict(x)
        return self.kmeans.predict(embedded_features)


