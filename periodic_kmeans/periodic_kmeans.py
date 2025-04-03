import numpy as np
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric, type_metric

from .periodic_average import periodic_average_2d


class PeriodicKMeans(kmeans):

    def __init__(self, data, period = 1, initial_centers = None, no_of_clusters = None, random_state = None):
        self.period = period
        self.period_2 = period / 2
        _centers = kmeans_plusplus_initializer(data, no_of_clusters, random_state = random_state).initialize() if initial_centers is None else initial_centers
        _metric = distance_metric(type_metric.USER_DEFINED, func = self.periodic_euclidean_distance)
        super().__init__(data, _centers, metric = _metric)


    def periodic_euclidean_distance(self, X: np.ndarray[float], Y: np.ndarray[float]): # distance between X and Y
        X_Y_wrapped = (X - Y + self.period_2) % self.period - self.period_2 # wrapping giving the smallest absolute difference in each coordinate
        return np.sqrt(np.sum(X_Y_wrapped**2))


    def __update_centers(self):
        """!
        @brief Calculate centers of clusters in line with contained objects.
        
        @return (numpy.array) Updated centers.
        
        """
        
        dimension = self.__pointer_data.shape[1]
        centers = np.zeros((len(self.__clusters), dimension))
        
        for index in range(len(self.__clusters)):
            cluster_points = self.__pointer_data[self.__clusters[index], :]
            centers[index] = periodic_average_2d(cluster_points, axis = 0, period = self.period)

        return np.array(centers)