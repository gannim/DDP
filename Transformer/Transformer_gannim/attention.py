# -*- coding: utf-8 -*-
# paper : https://arxiv.org/abs/1706.03762
import tensorflow as tf
import sys

import numpy as np

class Attention(object):
    def multi_head_attention(self, query, key, value, heads, masked=False):
        with tf.variable_scope("multihead_attention", reuse=tf.AUTO_REUSE):
            q_dims = query.get_shape().as_list()[-1]
            ## linear
            rep_q = tf.layers.dense(query, q_dims, activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name='rep_q')
            rep_k = tf.layers.dense(key, q_dims, activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name='rep_k')
            rep_v = tf.layers.dense(value, q_dims, activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name='rep_v')
            ## split heads
            split_q = tf.concat(tf.split(rep_q, heads, axis=-1), axis=0)
            split_k = tf.concat(tf.split(rep_k, heads, axis=-1), axis=0)
            split_v = tf.concat(tf.split(rep_v, heads, axis=-1), axis=0)
            ## attend heads
            att_map = self.scale_dot_product_attention(split_q, split_k, split_v, masked)
            concat_heads = tf.concat(tf.split(att_map, heads, axis=0), axis=-1)
            ## linear
            return tf.layers.dense(concat_heads, q_dims, activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name='Wo')
        
    @staticmethod
    def scale_dot_product_attention(query, key, value, masked=False):
        k_seq_len = float(key.get_shape().as_list()[-2])
        trans_k = tf.transpose(key, [0,2,1])
        outputs = tf.matmul(query, trans_k) / tf.sqrt(k_seq_len)
        if masked is True:
            diag_vals = tf.ones_like(outputs[0, :, :])
            #tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])
            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)
        att_map = tf.nn.softmax(outputs)
        return tf.matmul(att_map, value)

