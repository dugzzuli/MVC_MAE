import tensorflow as tf
import os
import pickle

from Utils.utils import *
from tensorflow.contrib.layers import l2_regularizer

w_init = lambda:tf.random_normal_initializer(stddev=0.02)

class Model(object):

    def __init__(self, config):
        self.config = config
        self.View1 = config['View']

        self.is_init = config['is_init']
        self.pretrain_params_path = config['pretrain_params_path']

        self.num_View1_layers = len(self.View1)

        weight_decay = self.config["weight_decay"]  # (1)定义weight_decay
        self.l2_reg = tf.contrib.layers.l2_regularizer(weight_decay)  # (2)定义l2_regularizer()


        if self.is_init:
            if os.path.isfile(self.pretrain_params_path):
                with open(self.pretrain_params_path, 'rb') as handle:
                    self.W_init, self.b_init = pickle.load(handle)


    def forward(self, x, drop_prob,view, reuse=False):
        
        print("=====forward_V"+view+"=====")
        self.View1_input_dim = x.shape[1]
        with tf.variable_scope('V'+view+'_encoder', reuse=reuse) as scope:
            cur_input = x
            # cur_input = gaussian_noise_layer(cur_input, 0.05)
            print(cur_input.get_shape())

            # cur_input, QK2 = NodeAtt(cur_input, self.View1_input_dim)

            # ============encoder===========
            struct = self.View1
            for i in range(self.num_View1_layers):
                name = 'V'+view+'_encoder' + str(i)
                if self.is_init:
                    print('V'+view+'_encoder:=====cur_input=====:'+str(i))
                    cur_input = tf.layers.dense(cur_input, units=struct[i],
                                                kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                                bias_initializer=tf.constant_initializer(self.b_init[name]),
                                                kernel_regularizer=self.l2_reg)
                else:
                    cur_input = tf.layers.dense(cur_input, units=struct[i], kernel_initializer=w_init(),
                                                kernel_regularizer=self.l2_reg
                                                )
                if i < self.num_View1_layers - 1:
                    cur_input = lrelu(cur_input)
                    cur_input = tf.layers.dropout(cur_input, drop_prob)
                print(cur_input.get_shape())

            net_H = cur_input

            # ====================decoder=============
            struct.reverse()
            cur_input = net_H
            for i in range(self.num_View1_layers - 1):
                name = 'V'+view+'_decoder' + str(i)
                if self.is_init:
                    print('V'+view+'_decoder:=====cur_input=====:' + str(i))
                    cur_input = tf.layers.dense(cur_input, units=struct[i + 1],
                                                kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                                bias_initializer=tf.constant_initializer(self.b_init[name]),
                                                kernel_regularizer=self.l2_reg)
                else:
                    cur_input = tf.layers.dense(cur_input, units=struct[i + 1], kernel_initializer=w_init(),
                                                kernel_regularizer=self.l2_reg
                                                )
                cur_input = lrelu(cur_input)
                cur_input = tf.layers.dropout(cur_input, drop_prob)
                print(cur_input.get_shape())

            name = 'V'+view+'_decoder' + str(self.num_View1_layers - 1)
            if self.is_init:
                cur_input = tf.layers.dense(cur_input, units=self.View1_input_dim,
                                            kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                            bias_initializer=tf.constant_initializer(self.b_init[name]),
                                            kernel_regularizer=self.l2_reg
                                            )
            else:
                cur_input = tf.layers.dense(cur_input, units=self.View1_input_dim, kernel_initializer=w_init(),
                                            kernel_regularizer=self.l2_reg
                                            )
            # cur_input = tf.nn.sigmoid(cur_input)
            # cur_input=tf.multiply(cur_input,QK2)
            x_recon = cur_input
            print(cur_input.get_shape())

            self.View1.reverse()
        print("=====forward_V"+view+"------end=====")
        return net_H, x_recon


class ModelUnet(object):

    def __init__(self, config):
        self.config = config
        self.View1 = config['View']

        self.is_init = config['is_init']
        self.pretrain_params_path = config['pretrain_params_path']

        self.num_View1_layers = len(self.View1)

        weight_decay = self.config["weight_decay"]  # (1)定义weight_decay
        self.l2_reg = tf.contrib.layers.l2_regularizer(weight_decay)  # (2)定义l2_regularizer()

        if self.is_init:
            if os.path.isfile(self.pretrain_params_path):
                with open(self.pretrain_params_path, 'rb') as handle:
                    self.W_init, self.b_init = pickle.load(handle)

    def forward(self, x, drop_prob, view, reuse=False):

        print("=====forward_V" + view + "=====")
        self.View1_input_dim = x.shape[1]
        with tf.variable_scope('V' + view + '_encoder', reuse=reuse) as scope:
            cur_input = x
            # cur_input = gaussian_noise_layer(cur_input, 0.05)
            print(cur_input.get_shape())

            # cur_input, QK2 = NodeAtt(cur_input, self.View1_input_dim)

            # ============encoder===========
            struct = self.View1
            skipC = []
            for i in range(self.num_View1_layers):
                name = 'V' + view + '_encoder' + str(i)
                if self.is_init:
                    print('V' + view + '_encoder:=====cur_input=====:' + str(i))
                    cur_input = tf.layers.dense(cur_input, units=struct[i],
                                                kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                                bias_initializer=tf.constant_initializer(self.b_init[name]),
                                                kernel_regularizer=self.l2_reg)
                else:
                    cur_input = tf.layers.dense(cur_input, units=struct[i], kernel_initializer=w_init(),
                                                kernel_regularizer=self.l2_reg
                                                )
                if i < self.num_View1_layers - 1:
                    cur_input = lrelu(cur_input)
                    cur_input = tf.layers.dropout(cur_input, drop_prob)
                print(cur_input.get_shape())

                skipC.append(cur_input)

            net_H = cur_input
            skipC[-1]=tf.zeros_like(skipC[-1])
            # ====================decoder=============
            struct.reverse()
            skipC.reverse()
            cur_input = net_H


            for i in range(self.num_View1_layers - 1):
                name = 'V' + view + '_decoder' + str(i)
                if self.is_init:
                    print('V' + view + '_decoder:=====cur_input=====:' + str(i))
                    cur_input = tf.layers.dense(cur_input+skipC[i], units=struct[i + 1],
                                                kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                                bias_initializer=tf.constant_initializer(self.b_init[name]),
                                                kernel_regularizer=self.l2_reg)
                else:
                    cur_input = tf.layers.dense(cur_input, units=struct[i + 1], kernel_initializer=w_init(),
                                                kernel_regularizer=self.l2_reg
                                                )
                cur_input = lrelu(cur_input)
                cur_input = tf.layers.dropout(cur_input, drop_prob)
                print(cur_input.get_shape())


            name = 'V' + view + '_decoder' + str(self.num_View1_layers - 1)
            if self.is_init:
                cur_input = tf.layers.dense(cur_input, units=self.View1_input_dim,
                                            kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                            bias_initializer=tf.constant_initializer(self.b_init[name]),
                                            kernel_regularizer=self.l2_reg
                                            )
            else:
                cur_input = tf.layers.dense(cur_input, units=self.View1_input_dim, kernel_initializer=w_init(),
                                            kernel_regularizer=self.l2_reg
                                            )
            x_recon = cur_input
            print(cur_input.get_shape())

            self.View1.reverse()
        print("=====forward_V" + view + "------end=====")
        return net_H, x_recon



def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise