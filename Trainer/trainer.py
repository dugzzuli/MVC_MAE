import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from Utils.utils import *
from Model.SingleAE import SingleAE
import pickle
from DEC.dec import  DEC2DUG


class Trainer(object):
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.drop_prob = config['drop_prob']
        self.beta = config['beta']
        self.gamma = config['gamma']
        self.alpha = config['alpha']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.model_path = config['model_path']
        self.beta_W = config['beta_W']
        self.View = config['View']
        self.L = config['L']
        self.View_num = config['View_num']
        self.cluster_num = config['cluster_num']
        self.dims = config['dims']

        self.params = {
            "encoder_dims": self.View[-1] * self.View_num,
            "n_clusters": self.cluster_num,
            "alpha": 1.0,
            'L': self.L,
            'batch_size': self.batch_size
        }

        self.xList = []
        for i in range(self.View_num):
            self.xList.append(tf.placeholder(tf.float32, [None, self.dims[i]], name='V' + str(i + 1)))
        self.adjs = []
        for i in range(self.View_num):
            self.adjs.append(tf.placeholder(tf.float32, [None, None], name='W' + str(i + 1)))

        self.mvList = self.model.getModel()
        self.optimizer, self.loss, self.dec = self._build_training_graph()
        self.net_H_List, self.H = self._build_eval_graph()

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def kl_divergence(self, p, p_hat):
        return tf.reduce_mean(p * tf.log(tf.clip_by_value(p, 1e-8, tf.reduce_max(p)))
                              - p * tf.log(tf.clip_by_value(p_hat, 1e-8, tf.reduce_max(p_hat)))
                              + (1 - p) * tf.log(tf.clip_by_value(1 - p, 1e-8, tf.reduce_max(1 - p)))
                              - (1 - p) * tf.log(tf.clip_by_value(1 - p_hat, 1e-8, tf.reduce_max(1 - p_hat))))

    def get_2nd_loss(self, X, newX, beta=5):
        TFMat = self.calc_sig(X)
        TFMat = X
        B = TFMat * (beta - 1) + 1
        return tf.reduce_sum(tf.pow((newX - X) * B, 2), 1)

    def calc_sig(self, DD):
        one = tf.ones_like(DD)
        zero = tf.zeros_like(DD)

        # 如果大于kth则为1，否则为0
        TFMat = tf.where(DD <= 0, x=zero, y=one)
        return TFMat;

    def get_1st_loss(self, H, adj_mini_batch):
        D = tf.diag(tf.reduce_sum(adj_mini_batch, 1))
        L = D - adj_mini_batch  ## L is laplation-matriX
        return 2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(H), L), H))

    def _build_training_graph(self):
        netList = []
        reconList = []
        # neg_netList = []
        # neg_reconList = []

        for i in range(self.View_num):
            net_V1, V1_recon = self.mvList[i].forward(self.xList[i], drop_prob=self.drop_prob, view=str(i + 1),
                                                      reuse=False)
            netList.append(net_V1)
            reconList.append(V1_recon)

        HDUG = tf.concat([tf.nn.l2_normalize(net, dim=1) for net in netList], axis=1)
        # HDUG = tf.concat([net for net in netList], axis=1)

        self.params['HDUG'] = HDUG
        dec = DEC2DUG(self.params)
        # ================high-order proximity & semantic proximity=============
        loss = 0
        rescontructloss = 0
        if np.sum(self.beta_W) == 0:
            print("非权重模式")
            for i in range(self.View_num):
                temploss = tf.reduce_mean(tf.reduce_sum(tf.square(self.xList[i] - reconList[i]), 1))
                # neg_temploss = tf.reduce_mean(tf.reduce_sum(tf.square(self.neg_xList[i] - neg_reconList[i]), 1))
                # temploss = tf.sqrt(temploss)
                rescontructloss = rescontructloss + temploss #+ neg_temploss
        else:
            print("权重模式")
            for i in range(self.View_num):
                print(self.beta_W)

                if (self.beta_W[i] == 0):
                    temploss = tf.reduce_mean(tf.reduce_sum(tf.square(self.xList[i] - reconList[i]), 1))
                else:
                    temploss = tf.reduce_mean(self.get_2nd_loss(self.xList[i], reconList[i], self.beta_W[i]))
                # temploss=tf.sqrt(temploss)
                rescontructloss += temploss

        recon_loss = rescontructloss
        # ===============cross modality proximity==================
        cross_modality_proximit = 0
        for i in range(self.View_num):
            for j in range(i + 1, self.View_num):
                # if(i==j):
                #     continue
                pre_logit_pos_single = tf.reduce_sum(tf.multiply(netList[i], netList[j]), 1)

                pos_loss_single = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pre_logit_pos_single),
                                                                          logits=pre_logit_pos_single)

                cross_modality_proximit = cross_modality_proximit + tf.reduce_mean(pos_loss_single)

        cross_modal_loss = cross_modality_proximit
        # =============== first-order proximity================
        first_order_loss = 0
        for i in range(self.View_num):
            pre_logit_pp_single = tf.matmul(netList[i], netList[i], transpose_b=True)

            pp_x_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.adjs[i] + tf.eye(tf.shape(self.adjs[i])[0]),
                                                                logits=pre_logit_pp_single) \
                        - tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(tf.diag_part(pre_logit_pp_single)),
                logits=tf.diag_part(pre_logit_pp_single))
            first_order_loss = first_order_loss + tf.reduce_mean(pp_x_loss)

        # first_order_loss = tf.reduce_mean(pp_x_loss + pp_z_loss  + pp_v3_loss )
        # ==========================================================
        loss = recon_loss * self.beta + cross_modal_loss * self.alpha + first_order_loss * self.gamma + dec.loss
        opt = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        return opt, loss, dec

    def _build_eval_graph(self):
        netList = []
        for i in range(self.View_num):
            net_V, _ = self.mvList[i].forward(self.xList[i], drop_prob=0.0, view=str(i + 1), reuse=True)
            netList.append(net_V)

        sent_outputs = tf.concat([tf.nn.l2_normalize(net, dim=1) for net in netList], axis=1)
        # sent_outputs = tf.concat([net for net in netList], axis=1)
        return netList, sent_outputs

    def getH(self, graph):
        train_emb = None
        train_label = None
        while True:
            mini_batch2 = graph.sample(self.batch_size, do_shuffle=False, with_label=True)

            feed_dict2 = {}
            for i in range(self.View_num):
                feed_dict2["V" + str(i + 1) + ":0"] = mini_batch2["V" + str(i + 1)]

            emb = self.sess.run(self.H, feed_dict=feed_dict2)

            if train_emb is None:
                train_emb = emb
                train_label = mini_batch2.Y
            else:
                train_emb = np.vstack((train_emb, emb))
                train_label = np.vstack((train_label, mini_batch2.Y))

            if graph.is_epoch_end:
                break
        return train_emb, train_label

    def train(self, graph):

        # initialize mu
        z, train_label = self.getH(graph)

        assign_mu_op = self.dec.get_assign_cluster_centers_op(z)
        _ = self.sess.run(assign_mu_op)

        for epoch in range(self.num_epochs):

            # q = self.sess.run(self.dec.q, feed_dict={self.v1: graph.V1,self.v2: graph.V2,self.v3:graph.V3})
            # p = self.dec.target_distribution(q)

            if (epoch % 100 == 0):
                z, train_label = self.getH(graph)
                assign_mu_op = self.dec.get_assign_cluster_centers_op(z)
                _ = self.sess.run(assign_mu_op)

                feed_dict1 = {}
                for i in range(self.View_num):
                    feed_dict1["V" + str(i + 1) + ":0"] = graph.ViewData[i]

                feed_dict1["dec_batch_size:0"] = np.shape(graph.ViewData[0])[0]

                q = self.sess.run(self.dec.q, feed_dict=feed_dict1)
                p = self.dec.target_distribution(q)

            # 初始化聚类中心

            idx1, neg_idx_List = self.generate_samples(graph)

            index = 0
            cost = 0.0
            cnt = 0
            while True:
                if index > graph.num_nodes:
                    break
                if index + self.batch_size < graph.num_nodes:
                    mini_batch1 = graph.sample_by_idx(idx1[index:index + self.batch_size])
                else:
                    mini_batch1 = graph.sample_by_idx(idx1[index:])

                feed_dict = {}
                for i in range(self.View_num):
                    feed_dict["V" + str(i + 1) + ":0"] = mini_batch1["V" + str(i + 1)]
                    feed_dict["W" + str(i + 1) + ":0"] = mini_batch1["W" + str(i + 1)]

                batch_p = p[mini_batch1.idx]
                feed_dict["P:0"] = batch_p
                feed_dict["dec_batch_size:0"] = np.shape(mini_batch1["V1"])[0]
                # 组织数据输入 feeddict

                index += self.batch_size

                loss, _ = self.sess.run([self.loss, self.optimizer],
                                        feed_dict=feed_dict)

                cost += loss
                cnt += 1

                if graph.is_epoch_end:
                    break
            cost /= cnt

            if epoch % 10 == 0:

                train_emb = None
                train_label = None
                while True:
                    mini_batch = graph.sample(self.batch_size, do_shuffle=False, with_label=True)

                    feed_dict3 = {}
                    for i in range(self.View_num):
                        feed_dict3["V" + str(i + 1) + ":0"] = mini_batch["V" + str(i + 1)]

                    batch_p = p[mini_batch.idx]
                    feed_dict3["P:0"] = np.squeeze(batch_p)
                    feed_dict3["dec_batch_size:0"] = np.shape(mini_batch["V1"])[0]

                    emb, embH = self.sess.run([self.dec.pred, self.H], feed_dict=feed_dict3)

                    emb = embH
                    if train_emb is None:
                        train_emb = emb
                        train_label = mini_batch.Y
                    else:
                        train_emb = np.vstack((train_emb, emb))
                        train_label = np.vstack((train_label, mini_batch.Y))

                    if graph.is_epoch_end:
                        break

                acc, nmi = node_clustering(train_emb, train_label)

                print('Epoch-{},loss: {:.4f}, acc {:.4f}, nmi {:.4f}'.format(epoch, cost, acc, nmi))

        self.save_model()

    def inferClusterDug(self, graph):
        self.sess.run(tf.global_variables_initializer())
        self.restore_model()
        print("Model restored from file: %s" % self.model_path)

        z, train_label = self.getH(graph)
        assign_mu_op = self.dec.get_assign_cluster_centers_op(z)
        _ = self.sess.run(assign_mu_op)
        feed_dict1 = {}
        for i in range(self.View_num):
            feed_dict1["V" + str(i + 1) + ":0"] = graph.ViewData[i]
        feed_dict1["dec_batch_size:0"] = np.shape(graph.ViewData[0])[0]
        q = self.sess.run(self.dec.q, feed_dict=feed_dict1)
        p = self.dec.target_distribution(q)

        train_emb = None
        train_label = None
        while True:
            mini_batch = graph.sample(np.shape(graph.ViewData[0])[0], do_shuffle=False, with_label=True)

            feed_dict = {}
            for i in range(self.View_num):
                feed_dict["V" + str(i + 1) + ":0"] = mini_batch["V" + str(i + 1)]

            batch_p = p[mini_batch.idx]
            feed_dict["P:0"] = np.squeeze(batch_p)
            feed_dict["dec_batch_size:0"] = np.shape(mini_batch["V1"])[0]

            emb = self.sess.run(self.dec.pred, feed_dict=feed_dict)

            if train_emb is None:
                train_emb = emb
                train_label = mini_batch.Y
            else:
                train_emb = np.vstack((train_emb, emb))
                train_label = np.vstack((train_label, mini_batch.Y))

            if graph.is_epoch_end:
                break

        acc, nmi = node_clusteringDug(train_emb, train_label)

        print('dug-dec acc {:.4f}, nmi {:.4f}'.format(acc, nmi))
        return acc, nmi

    def inferCluster(self, graph):
        self.sess.run(tf.global_variables_initializer())
        self.restore_model()
        print("Model restored from file: %s" % self.model_path)

        train_emb = None
        train_label = None
        while True:
            mini_batch = graph.sample(self.batch_size, do_shuffle=False, with_label=True)

            feed_dict = {}
            for i in range(self.View_num):
                feed_dict["V" + str(i + 1) + ":0"] = mini_batch["V" + str(i + 1)]
            emb = self.sess.run(self.H, feed_dict=feed_dict)

            if train_emb is None:
                train_emb = emb
                train_label = mini_batch.Y
            else:
                train_emb = np.vstack((train_emb, emb))
                train_label = np.vstack((train_label, mini_batch.Y))

            if graph.is_epoch_end:
                break

        acc, nmi = node_clustering(train_emb, train_label)

        print(' acc {:.4f}, nmi {:.4f}'.format(acc, nmi))
        return acc, nmi

    def generate_samples(self, graph):

        order = np.arange(graph.num_nodes)
        np.random.shuffle(order)
        net_H_List = None
        # index = 0
        # while True:
        #     if index > graph.num_nodes:
        #         break
        #     if index + self.batch_size < graph.num_nodes:
        #         mini_batch = graph.sample_by_idx(order[index:index + self.batch_size])
        #     else:
        #         mini_batch = graph.sample_by_idx(order[index:])
        #     index += self.batch_size
        #
        #     feed_dict1 = {}
        #     for i in range(self.View_num):
        #         feed_dict1["V" + str(i + 1) + ":0"] = graph.ViewData[i]
        #
        #
        #     net_H = self.sess.run(self.net_H_List,feed_dict=feed_dict1)
        #     if(net_H_List!=None):
        #         net_H_List=net_H
        #     else:
        #         net_H_List = net_H

        feed_dict1 = {}
        for i in range(self.View_num):
            feed_dict1["V" + str(i + 1) + ":0"] = graph.ViewData[i][order]
        net_H_List = self.sess.run(self.net_H_List, feed_dict=feed_dict1)

        neg_idx_List = []
        for i in range(self.View_num):
            for j in range(i + 1, self.View_num):
                X = net_H_List[i]
                Z = net_H_List[j]

                X = np.array(X)
                Z = np.array(Z)

                X = preprocessing.normalize(X, norm='l2')
                Z = preprocessing.normalize(Z, norm='l2')

                sim = np.dot(X, Z.T)
                neg_idx = np.argmin(sim, axis=1)
                neg_idx_List.append(neg_idx)

        return order, neg_idx_List[0]

    def save_model(self):
        self.saver.save(self.sess, self.model_path)

    def restore_model(self):
        self.saver.restore(self.sess, self.model_path)

