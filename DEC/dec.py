import tensorflow as tf
from sklearn.cluster import KMeans
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

from Utils.wkmeans import WKMeans


class DEC2DUG(object):
    def __init__(self, params):
        self.n_cluster = params["n_clusters"]
        self.kmeans = KMeans(n_clusters=params["n_clusters"], init="k-means++", n_init=20)
        self.H = params["HDUG"]
        self.alpha = params['alpha']

        self.batch_size = tf.placeholder(tf.int32, shape=(), name="dec_batch_size")
        # self.batch_size=params['batch_size']
        self.L = params['L']

        self.p = tf.placeholder(tf.float32, shape=(None, self.n_cluster), name="P")

        with tf.name_scope("distribution"):
            self.z = self.H
            self.mu = tf.Variable(tf.zeros(shape=(params["n_clusters"], params["encoder_dims"])), name="mu")
            self.q = self._soft_assignment(self.z, self.mu)

            self.pred = tf.argmax(self.q, axis=1)

            self.loss = self._kl_divergence(self.p, self.q) * self.L

    def get_assign_cluster_centers_op(self, features):
        # init mu
        print("Kmeans train start.")
        kmeans =self.kmeans.fit(np.nan_to_num(features))
        # kmeans = self.kmeans.fit(features)
        print("Kmeans train end.")
        return tf.assign(self.mu, kmeans.cluster_centers_)

    def _soft_assignment(self, embeddings, cluster_centers):
        """Implemented a soft assignment as the  probability of assigning sample i to cluster j.

        Args:
            embeddings: (num_points, dim)
            cluster_centers: (num_cluster, dim)

        Return:
            q_i_j: (num_points, num_cluster)
        """

        def _pairwise_euclidean_distance(a, b):
            p1 = tf.matmul(
                tf.expand_dims(tf.reduce_sum(tf.square(a), 1), 1),
                tf.ones(shape=(1, self.n_cluster))
            )
            p2 = tf.transpose(tf.matmul(
                tf.reshape(tf.reduce_sum(tf.square(b), 1), shape=[-1, 1]),
                tf.ones(shape=(self.batch_size, 1)),
                transpose_b=True
            ))
            res = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(a, b, transpose_b=True))
            return res

        dist = _pairwise_euclidean_distance(embeddings, cluster_centers)
        q = 1.0 / (1.0 + dist ** 2 / self.alpha) ** ((self.alpha + 1.0) / 2.0)
        q = (q / tf.reduce_sum(q, axis=1, keepdims=True))
        return q

    def target_distribution(self, q):
        p = q ** 2 / q.sum(axis=0)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    def _kl_divergence(self, target, pred):
        return tf.reduce_mean(tf.reduce_sum(target * tf.log(target / (pred)), axis=1))


class DEC2DUGFeat(object):
    def __init__(self, params):
        self.n_cluster = params["n_clusters"]
        self.kmeans = WKMeans(n_clusters=self.n_cluster,belta=2)
        self.H = params["HDUG"]
        self.alpha = params['alpha']

        self.batch_size = tf.placeholder(tf.int32, shape=(), name="dec_batch_size")
        # self.batch_size=params['batch_size']
        self.L = params['L']

        self.p = tf.placeholder(tf.float32, shape=(None, self.n_cluster), name="P")

        with tf.name_scope("distribution"):
            self.z = self.H
            self.mu = tf.Variable(tf.zeros(shape=(params["n_clusters"], params["encoder_dims"])), name="mu")
            self.q = self._soft_assignment(self.z, self.mu)

            self.pred = tf.argmax(self.q, axis=1)

            self.loss = self._kl_divergence(self.p, self.q) * self.L

    def get_assign_cluster_centers_op(self, features):
        # init mu
        print("Kmeans train start.")
        while True:
            self.kmeans.fit_predict(features)
            if self.kmeans.isConverge == True:
                # kmeans = self.kmeans.fit(features)
                print("Kmeans train end.")
                break
        return tf.assign(self.mu, self.kmeans.best_centers)

    def _soft_assignment(self, embeddings, cluster_centers):
        """Implemented a soft assignment as the  probability of assigning sample i to cluster j.

        Args:
            embeddings: (num_points, dim)
            cluster_centers: (num_cluster, dim)

        Return:
            q_i_j: (num_points, num_cluster)
        """

        def _pairwise_euclidean_distance(a, b):
            p1 = tf.matmul(
                tf.expand_dims(tf.reduce_sum(tf.square(a), 1), 1),
                tf.ones(shape=(1, self.n_cluster))
            )
            p2 = tf.transpose(tf.matmul(
                tf.reshape(tf.reduce_sum(tf.square(b), 1), shape=[-1, 1]),
                tf.ones(shape=(self.batch_size, 1)),
                transpose_b=True
            ))
            res = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(a, b, transpose_b=True))
            return res

        dist = _pairwise_euclidean_distance(embeddings, cluster_centers)
        q = 1.0 / (1.0 + dist ** 2 / self.alpha) ** ((self.alpha + 1.0) / 2.0)
        q = (q / tf.reduce_sum(q, axis=1, keepdims=True))
        return q

    def target_distribution(self, q):
        p = q ** 2 / q.sum(axis=0)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    def _kl_divergence(self, target, pred):
        return tf.reduce_mean(tf.reduce_sum(target * tf.log(target / (pred)), axis=1))