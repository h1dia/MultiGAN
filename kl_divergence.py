from scipy import stats
import numpy as np


def kl_divergence_2d(first_sample, second_sample, resolution=0.05):
    kde = stats.kde.gaussian_kde(first_sample)
    kde2 = stats.kde.gaussian_kde(second_sample)

    # make mesh
    X, Y = np.mgrid[-1:1 + resolution:resolution, -1:1 + resolution:resolution]
    xy = np.vstack((X.flatten(), Y.flatten())).T

    # calc probability
    list_kde = np.array([])
    list_kde2 = np.array([])
    for i in xy:
        list_kde = np.append(list_kde, kde(i))
        list_kde2 = np.append(list_kde2, kde2(i))

    # reduce inf
    list_kde[list_kde < 1.0e-300] = 1.0e-300
    list_kde2[list_kde2 < 1.0e-300] = 1.0e-300

    # return KL divergence
    return stats.entropy(list_kde, list_kde2)
