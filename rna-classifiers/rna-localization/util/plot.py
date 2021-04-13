import numpy as np

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from scipy import interp
from sklearn.metrics import roc_curve, auc, roc_auc_score

def histogram(data, bins, fileName):
    n, bins, patches = plt.hist(data, bins, histtype='bar', stacked=True)
    plt.grid(True)

    plt.savefig('img/'+fileName+'.png')
    plt.close()

def roc(false_pos_rate, true_pos_rate, fileName):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    # Labels
    for k in false_pos_rate:
        fpr = false_pos_rate[k]
        tpr = true_pos_rate[k]
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=' %s (AUC = %0.2f)' % (k, roc_auc), lw=1)

        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
    
    # Mean
    mean_tpr /= len(false_pos_rate)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean (AUC = %0.2f)' % mean_auc, lw=2)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('img/'+fileName+'.png')
    plt.close()