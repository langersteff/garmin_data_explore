import numpy as np
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn import metrics

class ClustererBase:

    def __init__(self, print_summary=False, print_unique_pred=True, plot_pred=False, save_pred=True):
        super().__init__()
        self.print_summary = print_summary
        self.print_unique_pred = print_unique_pred
        self.plot_pred = plot_pred
        self.save_pred = save_pred

    def fit_predict(self, data):
        raise Exception("TODO: Override")

    def set_prefixes(self, prefix, num_clusters, window_length, sub_sample_length, eval_prefix):
        self.prefix = prefix
        self.eval_prefix = eval_prefix
        self.title = str.capitalize(eval_prefix)
        self.data_prefix = "%s_%s_%s" % (prefix, str(window_length), str(sub_sample_length))
        self.experiment_prefix = "%s_%s" % (self.data_prefix, str(num_clusters))
        self.filename_y_pred = "evaluation/%s_%s_y_pred" % (self.experiment_prefix, eval_prefix)
        self.filename_score = "evaluation/%s_%s_score.csv" % (self.experiment_prefix, eval_prefix)

    def plot_y_pred(self, y_pred, latitudes, longitudes, title=None):
        if title is not None:
            self.title = title
        print(self.title)
        print("y", np.unique(y_pred, return_counts=True))
        geometry = gpd.points_from_xy(latitudes, longitudes)
        fig = figure(1, figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
        gdf = GeoDataFrame(geometry=geometry)
        gdf.plot(c=y_pred, figsize=(20, 30))
        plt.show()
        return fig

    def save_scores(self, y_true, y_pred, data):

        if len(data.shape) > 2:
            data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))

        scores = []
        scores.append(['score_name', 'score_result'])
        scores.append(['adjusted_rand_score', metrics.adjusted_rand_score(y_true, y_pred)])
        scores.append(['adjusted_mutual_info_score', metrics.adjusted_mutual_info_score(y_true, y_pred)])
        scores.append(['homogeneity_score', metrics.homogeneity_score(y_true, y_pred)])
        scores.append(['completeness_score', metrics.completeness_score(y_true, y_pred)])
        scores.append(['v_measure_score', metrics.v_measure_score(y_true, y_pred)])
        scores.append(['fowlkes_mallows_score', metrics.fowlkes_mallows_score(y_true, y_pred)])

        if len(np.unique(y_pred, return_counts=True)[0]) > 1:
            scores.append(['silhouette_score', metrics.silhouette_score(data, y_pred)])
            scores.append(['davies_bouldin_score', metrics.davies_bouldin_score(data, y_pred)])
            scores.append(['calinski_harabasz_score', metrics.calinski_harabasz_score(data, y_pred)])
        else:
            scores.append(['silhouette_score', 0])
            scores.append(['davies_bouldin_score', 0])
            scores.append(['calinski_harabasz_score', 0])

        print(scores)

        np.savetxt(self.filename_score, scores, fmt='%s,%s', delimiter=",")