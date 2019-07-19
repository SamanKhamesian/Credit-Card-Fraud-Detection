import numpy as np
from sklearn.cluster import AgglomerativeClustering as Linkage
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


# Kmeans clustering algorithm
class KMeansClustering:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.__model = KMeans(n_clusters=n_clusters, random_state=0)

    def run(self, data):
        print('Clustering ...')
        data = np.array([[x] for x in data])
        print('Clustering is finished.')
        return self.__model.fit(X=data).labels_

    def predict(self, sample):
        return self.__model.predict(X=[[sample]])


# DBSCAN clustering algorithm
class DBSCANClustering:
    def __init__(self, n_neighbors=5, eps=5):
        self.__model = DBSCAN(min_samples=n_neighbors, eps=eps)

    def run(self, data):
        print('Clustering ...')
        data = np.array([[x] for x in data])
        print('Clustering is finished.')
        return self.__model.fit(X=data).labels_ + 1

    def predict(self, sample):
        return self.__model.fit_predict(X=[[sample]]) + 1


# The single link method clustering
class SLINKClustering:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.__model = Linkage(linkage='single', n_clusters=n_clusters)

    def run(self, data):
        print('Clustering ...')
        data = np.array([[x] for x in data])
        print('Clustering is finished.')
        return self.__model.fit(X=data).labels_

    def predict(self, sample):
        x = [[sample]]
        for i in range(self.n_clusters):
            x.append([sample + i])
        return self.__model.fit_predict(X=x)[0]


# The complete link method clustering
class CLINKClustering:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.__model = Linkage(linkage='complete', n_clusters=n_clusters)

    def run(self, data):
        print('Clustering ...')
        data = np.array([[x] for x in data])
        print('Clustering is finished.')
        return self.__model.fit(X=data).labels_

    def predict(self, sample):
        x = [[sample]]
        for i in range(self.n_clusters):
            x.append([sample + i])
        return self.__model.fit_predict(X=x)[0]


# The average method clustering
class AVGClustering:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.__model = Linkage(linkage='average', n_clusters=n_clusters)

    def run(self, data):
        print('Clustering ...')
        data = np.array([[x] for x in data])
        print('Clustering is finished.')
        return self.__model.fit(X=data).labels_

    def predict(self, sample):
        x = [[sample]]
        for i in range(self.n_clusters):
            x.append([sample + i])
        return self.__model.fit_predict(X=x)[0]
