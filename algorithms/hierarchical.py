from itertools import combinations
import random
import networkx as nx
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from networkx.drawing.nx_pydot import graphviz_layout
from scipy.spatial import distance_matrix as matrix_distance

np.random.seed(0)
from matplotlib import pyplot as plt
import scipy.io
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_data(num_examples=500):
    fptr = open('cluster_data.csv', 'r')
    data = []
    for line in fptr.readlines():
        line = line.strip().split(',')
        data.append([int(line[0]), int(line[1])])

    data = random.sample(data, num_examples)
    data = np.array(data)
    return data


def normalize(data, num_features=100, reduce=True):
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    if not reduce:
        return data
    pca = PCA(n_components=num_features, random_state=22)
    pca.fit(data)
    x = pca.transform(data)
    return x


def create_distance_matrix(data):
    num_examples = data.shape[0]
    distance_matrix = np.zeros((num_examples, num_examples))
    for i, d1 in enumerate(data):
        for j, d2 in enumerate(data):
            distance_matrix[i][j] = np.linalg.norm(d1 - d2)
    return distance_matrix


def flatten(list_object, dtype=int):
    _list = []
    for elem in list_object:
        if isinstance(elem, list):
            elem = flatten(elem)
            _list += elem
        elif isinstance(elem, dtype):
            _list.append(elem)
    return _list


def text_to_image(text):
    fig = Figure(figsize=(5, 5))
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    ax.text(0.5, 0.5, text, fontsize=60)
    ax.axis('off')

    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(int(width), int(height), 3)
    image = Image.fromarray(image)
    return image


def distance_between_cluster(cluster_pair, distance_matrix, type_of_clustering):
    cluster1 = flatten(cluster_pair[0])
    cluster2 = flatten(cluster_pair[1])
    cross_cluster_distances = []
    for i in cluster1:
        for j in cluster2:
            cross_cluster_distances.append(distance_matrix[i][j])
    if type_of_clustering == 'single':
        return min(cross_cluster_distances)
    elif type_of_clustering == 'complete':
        return max(cross_cluster_distances)
    elif type_of_clustering == 'average':
        return np.mean(cross_cluster_distances)


def hierarchical_cluster(distance_matrix, type_of_clustering='single'):
    if type_of_clustering not in ['single', 'complete', 'average']:
        return []
    clusters = [[i] for i in range(len(distance_matrix))]
    while len(clusters) > 2:
        cluster_joins = [list(pair) for pair in combinations(clusters, 2)]
        cluster_distances = [distance_between_cluster(pair, distance_matrix, type_of_clustering)
                             for pair in cluster_joins]
        closest_pair_index = np.argmin(cluster_distances)
        closest_pair = cluster_joins[closest_pair_index]
        clusters = [cluster for cluster in clusters if cluster not in closest_pair]
        clusters.append(closest_pair)

    return clusters


def construct_tree(clusters, G, num=0):
    root = f"[{num}]"
    G.add_node(root)
    for child in clusters:
        if isinstance(child, int):
            child_node = f"({child})"
            G.add_node(child_node)
        else:
            child_node = f"[{num + 1}]"
            num = construct_tree(child, G, num + 1)
        G.add_edge(root, child_node)
    return num


def main():
    image = scipy.io.loadmat('../binaryalphadigs.mat')
    images = []
    feat = []
    labels = []
    for num in range(image['dat'].shape[0]):
        label = image['classlabels'][0][num][0]
        for img_i in range(image['dat'].shape[1]):
            img_arr = image['dat'][num][img_i]
            img_arr = np.uint8(img_arr * 255)
            img = Image.fromarray(img_arr, 'L')
            images.append(img)
            img_features = img_arr.flatten()
            feat.append(img_features)
            labels.append(label)
    feat = np.array(feat)
    data = normalize(feat, 5)
    num_examples = 20
    data = data[:num_examples]

    # data = np.array([[1.5, 2.0],
    #                  [3.0, 1.0],
    #                  [3.5, 2.5],
    #                  [1.0, 0.5],
    #                  [2.5, 2.0]])

    distance_matrix = matrix_distance(data, data)
    clusters = hierarchical_cluster(distance_matrix, 'single')
    print(clusters)
    G = nx.DiGraph()
    construct_tree(clusters, G)

    plt.title('draw_networkx')
    pos = graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, arrows=True)

    plt.savefig("tree.jpg")


def main2():
    data = get_data(20)
    data = normalize(data, reduce=False)

    # data = np.array([[1.5, 2.0],
    #                  [3.0, 1.0],
    #                  [3.5, 2.5],
    #                  [1.0, 0.5],
    #                  [2.5, 2.0]])

    distance_matrix = create_distance_matrix(data)
    clusters = hierarchical_cluster(distance_matrix, 'complete')
    print(clusters)
    G = nx.DiGraph()
    construct_tree(clusters, G)

    nx.nx_agraph.write_dot(G, 'test.dot')

    # same layout using matplotlib with no labels
    plt.title('draw_networkx')
    pos = graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, arrows=True)
    plt.savefig("tree_sample.jpg")


if __name__ == '__main__':
    main()
