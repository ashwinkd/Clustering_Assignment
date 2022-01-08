import random

# import cv2
import numpy as np
import scipy.io
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


# from sklearn.decomposition import PCA
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.models import Model

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# model = VGG16()
# model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
#
#
# def get_VGG_features(image):
#     image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
#     image = image.reshape(1, 224, 224, 1)
#     image = np.repeat(image, 3, -1)
#     imgx = preprocess_input(image)
#     features = model.predict(imgx, use_multiprocessing=True)
#     return features
#

def get_features(image):
    features = image.flatten()
    return features


image = scipy.io.loadmat('binaryalphadigs.mat')
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
        # img_features = get_VGG_features(img_arr)
        img_features = get_features(img_arr)
        feat.append(img_features)
        labels.append(label)

feat = np.array(feat)
# feat = feat.reshape(-1, 4096)
feat = feat.reshape(-1, 320)
unique_labels = list(set(labels))

# pca = PCA(n_components=100, random_state=22)
# pca.fit(feat)
# x = pca.transform(feat)
x = feat

kmeans = KMeans(n_clusters=len(unique_labels), n_jobs=-1, random_state=22)
kmeans.fit(x)

groups = {}
for img, cluster in zip(images, kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
    groups[cluster].append(img)


def view_cluster(cluster_label, cluster):
    fig = plt.figure(figsize=(25, 25))
    if len(cluster) > 30:
        cluster = random.sample(cluster, 30)
    for index, img_member in enumerate(cluster):
        ax = fig.add_subplot(6, 5, index + 1)
        ax.imshow(img_member)
        ax.axis('off')
    plt.savefig(f"flatten_result/{cluster_label}.jpg")


for cluster_label, cluster in groups.items():
    print(cluster_label)
    view_cluster(cluster_label, cluster)
