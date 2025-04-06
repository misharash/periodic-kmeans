import numpy
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric, type_metric

from .periodic_average import periodic_average_2d


class PeriodicKMeans(kmeans):

    def __init__(self, data, period = 1, initial_centers = None, no_of_clusters = None, random_state = None):
        self.period = period
        self.period_2 = period / 2
        _metric = distance_metric(type_metric.USER_DEFINED, func = self.periodic_euclidean_distance_square_numpy)
        _centers = kmeans_plusplus_initializer(data, no_of_clusters, metric = _metric, random_state = random_state).initialize() if initial_centers is None else initial_centers
        super().__init__(data, _centers, metric = _metric)


    def periodic_euclidean_distance_square_numpy(self, object1, object2, simple = True):
        """!
        @brief Calculate square Euclidean distance with periodicity between two objects using numpy.

        @param[in] object1 (array_like): The first array_like object.
        @param[in] object2 (array_like): The second array_like object.
        @param[in] simple (boolean): If False, compute the full distance matrix between all pairs in two sets of points.

        @return (double) Square Euclidean distance between two objects.

        """
        diff_wrapped = ((object1 - object2 if simple else object1[:, None, :] - object2[None, :, :]) + self.period_2) % self.period - self.period_2 # wrapping giving the smallest absolute difference in each coordinate
        return numpy.sum(numpy.square(diff_wrapped), axis=-1)


    def periodic_euclidean_distance_numpy(self, object1, object2, simple = True):
        """!
        @brief Calculate Euclidean distance with periodicity between two objects using numpy.

        @param[in] object1 (array_like): The first array_like object.
        @param[in] object2 (array_like): The second array_like object.
        @param[in] simple (boolean): If False, compute the full distance matrix between all pairs in two sets of points.

        @return (double) Euclidean distance between two objects.

        """
        diff_wrapped = ((object1 - object2 if simple else object1[:, None, :] - object2[None, :, :]) + self.period_2) % self.period - self.period_2 # wrapping giving the smallest absolute difference in each coordinate
        return numpy.sqrt(numpy.sum(numpy.square(diff_wrapped), axis=-1))


    def _kmeans__update_centers(self): # need to prepend parent class name to override this extra protected method
        """!
        @brief Calculate centers of clusters in line with contained objects.
        
        @return (numpy.array) Updated centers.
        
        """
        
        dimension = self._kmeans__pointer_data.shape[1]
        centers = numpy.zeros((len(self._kmeans__clusters), dimension))
        
        for index in range(len(self._kmeans__clusters)):
            cluster_points = self._kmeans__pointer_data[self._kmeans__clusters[index], :]
            centers[index] = periodic_average_2d(cluster_points, axis = 0, period = self.period)

        return numpy.array(centers)


    def predict(self, points):
        """!
        @brief Calculates the closest cluster to each point.

        @param[in] points (array_like): Points for which closest clusters are calculated.

        @return (list) List of closest clusters for each point. Each cluster is denoted by index. Return empty
                 collection if 'process()' method was not called.

        """

        nppoints = numpy.array(points)
        if len(self._kmeans__clusters) == 0:
            return []

        differences = self.periodic_euclidean_distance_square_numpy(nppoints, self._kmeans__centers, simple = False)

        return numpy.argmin(differences, axis=1)


    def _kmeans__calculate_dataset_difference(self, amount_clusters): # need to prepend parent class name to override this extra protected method
        """!
        @brief Calculate distance from each point to each cluster center.

        """
        return self.periodic_euclidean_distance_square_numpy(self._kmeans__centers[:amount_clusters], self._kmeans__pointer_data, simple = False)


    def _kmeans__calculate_changes(self, updated_centers): # need to prepend parent class name to override this extra protected method
        """!
        @brief Calculates changes estimation between previous and current iteration using centers for that purpose.

        @param[in] updated_centers (array_like): New cluster centers.

        @return (float) Maximum changes between centers.

        """
        if len(self._kmeans__centers) != len(updated_centers):
            maximum_change = float('inf')
        else:
            changes = self.periodic_euclidean_distance_square_numpy(self._kmeans__centers, updated_centers)
            maximum_change = numpy.max(changes)

        return maximum_change