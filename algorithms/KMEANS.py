import random

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)


def get_data(num_examples=500):
    fptr = open('cluster_data.csv', 'r')
    data = []
    for line in fptr.readlines():
        line = line.strip().split(',')
        data.append([int(line[0]), int(line[1])])

    data = random.sample(data, num_examples)
    data = np.array(data)
    return data


def get_init_centroid(data, k):
    init_centroid_i = random.sample(list(range(data.shape[0])), k)
    init_centroid = data[init_centroid_i]
    return init_centroid


def normalize(data):
    _mean = np.mean(data, axis=0)
    _std = np.std(data, axis=0)
    data = (data - _mean) / _std
    return data


def kmeans(data, init_centroids):
    original_clusters = [[] for i in range(len(init_centroids))]
    while True:
        clusters = [[] for i in range(len(init_centroids))]
        for i, point in enumerate(data):
            d = []
            for c in init_centroids:
                d.append(np.linalg.norm(point - c))
            clusters[np.argmin(d)] += [i]
        sse = 0
        for i in range(len(init_centroids)):
            c = np.mean([data[p] for p in clusters[i]], axis=0)
            init_centroids[i] = c
            c_dist = [np.linalg.norm(data[p] - c) ** 2 for p in clusters[i]]
            sse += np.sum(c_dist)
        # print(clusters, sse, init_centroids)
        if original_clusters == clusters:
            break
        else:
            # print(clusters, sse, init_centroids)
            original_clusters = clusters
    return original_clusters


def plot_clusters(data, clusters):
    num_examples = data.shape[0]
    color_map = {i: (random.randint(0, 255) / 255,
                     random.randint(0, 255) / 255,
                     random.randint(0, 255) / 255)
                 for i in range(len(clusters))}

    colors = [(0.8, 0.8, 0.8) for i in range(num_examples)]
    for i, cluster in enumerate(clusters):
        for cl in cluster:
            colors[cl] = color_map[i]

    x = [d[0] for d in data]
    y = [d[1] for d in data]

    plt.scatter(x=x, y=y, c=colors)
    plt.show()


def main():
    data = get_data()
    data = normalize(data)
    k = 15
    init_centroid = get_init_centroid(data, k)
    # data = np.array([[1.5, 2.0],
    #                  [3.0, 1.0],
    #                  [3.5, 2.5],
    #                  [1.0, 0.5],
    #                  [2.5, 2.0]])
    #
    # init_centroid = np.array([[3.0, 1.0],
    #                     [2.5, 2.0]])

    clusters = kmeans(data, init_centroid)
    plot_clusters(data, clusters)


if __name__ == '__main__':
    main()
