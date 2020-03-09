import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib.pyplot import figure


class MtbVisualizer:

    def print_confusion_matrix(self, y, y_pred, labels):
        cm = confusion_matrix(y, y_pred, labels, normalize='all')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')

        plt.show()