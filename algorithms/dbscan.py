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


def normalize(data):
    _mean = np.mean(data, axis=0)
    _std = np.std(data, axis=0)
    data = (data - _mean) / _std
    return data


def create_distance_matrix(data):
    num_examples = data.shape[0]
    distance_matrix = np.zeros((num_examples, num_examples))
    for i, d1 in enumerate(data):
        for j, d2 in enumerate(data):
            distance_matrix[i][j] = np.linalg.norm(d1 - d2)
    return distance_matrix


def add_labels(distance_matrix, eps, minPts):
    num_examples = distance_matrix.shape[0]
    labels = [(None, None) for l in range(num_examples)]
    for i in range(num_examples):
        in_range = {key: dist for key, dist in enumerate(distance_matrix[i]) if dist < eps}
        if len(in_range) > minPts:
            labels[i] = ('core', i)

    for i, l in enumerate(labels):
        if l[0] != 'core':
            in_range = {key: dist
                        for key, dist in enumerate(distance_matrix[i])
                        if dist < eps
                        and labels[key][0] == 'core'
                        and key != i}
            if not in_range:
                labels[i] = ('noise', None)
                continue
            closest = min(in_range, key=in_range.get)
            if closest is not None:
                labels[i] = ('border', closest)
    return labels


def create_clusters(labels, distance_matrix, eps):
    clusters = []
    for i, l in enumerate(labels):
        if l[0] == 'core':
            in_range = {key: dist
                        for key, dist in enumerate(distance_matrix[i])
                        if dist < eps
                        and labels[key][0] == 'core'}
            cluster = set(in_range.keys())
            if not clusters:
                clusters.append(cluster)
                continue
            common_cluster_i = None
            for cl, group in enumerate(clusters):
                if not group.isdisjoint(cluster):
                    common_cluster_i = cl
            if common_cluster_i is not None:
                clusters[common_cluster_i] = clusters[common_cluster_i].union(cluster)
            else:
                clusters.append(cluster)
    for i, l in enumerate(labels):
        if l[0] == 'border':
            in_range = {key: dist
                        for key, dist in enumerate(distance_matrix[i])
                        if dist < eps
                        and labels[key][0] == 'core'
                        and key != i}
            core = min(in_range, key=in_range.get)
            for j, cluster in enumerate(clusters):
                if core in cluster:
                    clusters[j].add(i)

    return clusters


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
    eps = 0.2
    minPts = 5

    data = get_data()
    data = normalize(data)
    # data = np.array([[1.5, 2.0],
    #                  [3.0, 1.0],
    #                  [3.5, 2.5],
    #                  [1.0, 0.5],
    #                  [2.5, 2.0]])
    distance_matrix = create_distance_matrix(data)
    labels = add_labels(distance_matrix, eps, minPts)
    clusters = create_clusters(labels, distance_matrix, eps)
    plot_clusters(data, clusters)


if __name__ == '__main__':
    main()
