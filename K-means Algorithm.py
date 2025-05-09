# Ganna Ahmed Abd El-Naby Silem
# 20210102
import random

import numpy as np
import pandas as pd


class KMeansAlgo:
    def __init__(self, sample_percentage, n_clusters, input_file):
        self.sample_percentage = sample_percentage
        self.n_clusters = n_clusters
        self.dataset = []
        self.processing_data = []
        self.max_iteration = 1100
        self.centroids = []
        self.file = input_file
        self.loadFile()
        self.preprocessing_data()

    def loadFile(self):
        customer_behavior_data = pd.read_csv(self.file)
        customer_behavior_data = customer_behavior_data.sample(frac=self.sample_percentage / 100, random_state=42)
        self.dataset = customer_behavior_data

    def preprocessing_data(self):
        data = self.dataset
        data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

        features = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']

        self.processing_data = data[features].values.tolist()

    def calculate_euclidean_distance(self, point, centroid):
        distance = 0
        for i in range(len(point)):
            distance += (point[i] - centroid[i]) ** 2
        return distance ** 0.5

    def update_centroid_value(self, clusters):
        new_centroids = []
        for i, cluster in enumerate(clusters):
            mean_result = [[] for _ in range(4)]
            for point in cluster:
                for index in range(len(point)):
                    mean_result[index].append(point[index])
            centroid = []
            for values in mean_result:
                sum_result = sum(values)
                if len(values) != 0:
                    centroid.append(round(sum_result / len(values)))

            if centroid:
                new_centroids.append(centroid)
            else:
                new_centroids.append(self.centroids[i])

        return new_centroids

    def algorithem_process(self):
        self.centroids = random.sample(self.processing_data, self.n_clusters)
        iteration = self.max_iteration
        for iteration in range(self.max_iteration):
            clusters = [[] for _ in range(self.n_clusters)]
            for point in self.processing_data:
                distances = []
                for centroid in self.centroids:
                    distance = self.calculate_euclidean_distance(point, centroid)
                    distances.append(distance)

                closest_centroid_index = 0
                min_distance = distances[0]
                for i in range(len(distances)):
                    if distances[i] < min_distance:
                        min_distance = distances[i]
                        closest_centroid_index = i

                clusters[closest_centroid_index].append(point)

            new_centroids = self.update_centroid_value(clusters)

            if new_centroids == self.centroids:
                break
            else:
                self.centroids = new_centroids

        return clusters

    def detect_outliers(self, clusters):
        outliers = []
        for cluster_index, cluster in enumerate(clusters):
            centroid = self.centroids[cluster_index]
            distances = []
            for point in cluster:
                distance = self.calculate_euclidean_distance(point, centroid)
                distances.append(distance)

            Q1 = np.percentile(distances, 25)
            Q3 = np.percentile(distances, 75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            for i, point in enumerate(cluster):
                if (distances[i] < lower_bound) or (distances[i] > upper_bound):
                    outliers.append((cluster_index, point))

        return outliers

    def get_customer_data(self, clusters, outliers):
        customer_clusters = []
        outlier_customers = []

        customer_ids = self.dataset['CustomerID'].values
        for cluster_idx, cluster in enumerate(clusters):
            for point_idx, point in enumerate(cluster):
                original_idx = self.processing_data.index(point)
                customer_id = customer_ids[original_idx]
                customer_clusters.append({
                    'ClusterID': int(cluster_idx),
                    'CustomerID': int(customer_id),
                    'Gender': point[0],
                    'Age': point[1],
                    'Annual Income': point[2],
                    'Spending Score': point[3]
                })

        for cluster_idx, point in outliers:
            original_idx = self.processing_data.index(point)
            customer_id = customer_ids[original_idx]
            outlier_customers.append({
                'CustomerID': int(customer_id),
                'Gender': point[0],
                'Age': point[1],
                'Annual Income': point[2],
                'Spending Score': point[3]
            })

        return customer_clusters, outlier_customers