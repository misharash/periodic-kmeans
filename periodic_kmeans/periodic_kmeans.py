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
        # enable some numpy vectorization for distance computations
        self._kmeans__metric.__numpy = True
        self._kmeans__metric._distance_metric__create_distance_calculator()


    def periodic_euclidean_distance_square_numpy(self, object1, object2):
        """!
        @brief Calculate square Euclidean distance with periodicity between two objects using numpy.

        @param[in] object1 (array_like): The first array_like object.
        @param[in] object2 (array_like): The second array_like object.

        @return (double) Square Euclidean distance between two objects.

        """
        diff_wrapped = (object1 - object2 + self.period_2) % self.period - self.period_2 # wrapping giving the smallest absolute difference in each coordinate
        if len(object1.shape) > 1 or len(object2.shape) > 1:
            return numpy.sum(numpy.square(diff_wrapped), axis=1).T # left the transposition as in pyclustering.utils.metric.euclidean_distance_square_numpy, not sure why it is there because the array should not become more than 2-dimensional
        else:
            return numpy.sum(numpy.square(diff_wrapped))


    def periodic_euclidean_distance_numpy(self, object1, object2):
        """!
        @brief Calculate Euclidean distance with periodicity between two objects using numpy.

        @param[in] object1 (array_like): The first array_like object.
        @param[in] object2 (array_like): The second array_like object.

        @return (double) Euclidean distance between two objects.

        """
        diff_wrapped = (object1 - object2 + self.period_2) % self.period - self.period_2 # wrapping giving the smallest absolute difference in each coordinate
        if len(object1.shape) > 1 or len(object2.shape) > 1:
            return numpy.sqrt(numpy.sum(numpy.square(diff_wrapped), axis=1))
        else:
            return numpy.sqrt(numpy.sum(numpy.square(diff_wrapped)))


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

        differences = numpy.zeros((len(nppoints), len(self._kmeans__centers)))
        for index_point in range(len(nppoints)):
            differences[index_point] = self._kmeans__metric(nppoints[index_point], self._kmeans__centers)

        return numpy.argmin(differences, axis=1)


    def _kmeans__calculate_dataset_difference(self, amount_clusters): # need to prepend parent class name to override this extra protected method
        """!
        @brief Calculate distance from each point to each cluster center.

        """
        dataset_differences = numpy.zeros((amount_clusters, len(self._kmeans__pointer_data)))
        for index_center in range(amount_clusters):
            dataset_differences[index_center] = self._kmeans__metric(self._kmeans__pointer_data, self._kmeans__centers[index_center])

        return dataset_differences


    def _kmeans__calculate_changes(self, updated_centers): # need to prepend parent class name to override this extra protected method
        """!
        @brief Calculates changes estimation between previous and current iteration using centers for that purpose.

        @param[in] updated_centers (array_like): New cluster centers.

        @return (float) Maximum changes between centers.

        """
        if len(self._kmeans__centers) != len(updated_centers):
            maximum_change = float('inf')
        else:
            changes = self._kmeans__metric(self._kmeans__centers, updated_centers)
            maximum_change = numpy.max(changes)

        return maximum_change