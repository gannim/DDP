# -*- coding: utf-8 -*-
# original code : https://github.com/changwookjun/Transformer
t
import tensorflow as tf
import sys

import numpy as np

class Transformer(object):
    @staticmethod
    def positional_encoding(dims, sentence_length, dtype=tf.float32):
        # https://arxiv.org/abs/1706.03762
        ary = []
        for i in range(dims):
            for pos in range(sentence_length):
                ary.append(pos/np.power(10000, 2*i/dims))
        encoded_vec = np.array(ary)
        encoded_vec[::2] = np.sin(encoded_vec[::2])
        encoded_vec[1::2] = np.cos(encoded_vec[1::2])
        return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dims]), dtype=dtype)

    @staticmethod
    def layer_norm(inputs, eps=1e-6):
        # LayerNorm(x + Sublayer(x))
        feature_shape = inputs.get_shape()[-1:]
        #  평균과 표준편차을 넘겨 준다.
        mean = tf.keras.backend.mean(inputs, [-1], keepdims=True)
        std = tf.keras.backend.std(inputs, [-1], keepdims=True)
        beta = tf.Variable(tf.zeros(feature_shape), trainable=False)
        gamma = tf.Variable(tf.ones(feature_shape), trainable=False)
    
        return gamma * (inputs - mean) / (std + eps) + beta

    def sublayer_connection(self, inputs, sublayer, dropout):
        # LayerNorm(x + Sublayer(x))
        return tf.layers.dropout(self.layer_norm(inputs + sublayer), dropout)
    @staticmethod
    def feed_forward(inputs, num_units, dropout):
        # FFN(x) = max(0, xW1 + b1)W2 + b2 
        with tf.variable_scope("feed_forward", reuse=tf.AUTO_REUSE):
            outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
            outputs = tf.layers.dropout(outputs, dropout)
            return tf.layers.dense(outputs, num_units[1])
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

    def multi_head_attention(self, query, key, value, heads, masked=False):
        with tf.variable_scope("multihead_attention", reuse=tf.AUTO_REUSE):
            q_dims = query.get_shape().as_list()[-1]
            rep_q = tf.layers.dense(query, q_dims, activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name='rep_q')
            rep_k = tf.layers.dense(key, q_dims, activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name='rep_k')
            rep_v = tf.layers.dense(value, q_dims, activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name='rep_v')

            split_q = tf.concat(tf.split(rep_q, heads, axis=-1), axis=0)
            split_k = tf.concat(tf.split(rep_k, heads, axis=-1), axis=0)
            split_v = tf.concat(tf.split(rep_v, heads, axis=-1), axis=0)

            att_map = self.scale_dot_product_attention(split_q, split_k, split_v, masked)
            return tf.concat(tf.split(att_map, heads, axis=0), axis=-1)
        
    def encoder_module(self, inputs, num_units, heads, dropout):
        multi_head_att = self.multi_head_attention(inputs, inputs, inputs, heads)
        self_att = self.sublayer_connection(inputs, multi_head_att, dropout)
        ## conv_1d_layer or fnn 
        network_layer = self.feed_forward(self_att, num_units, dropout)
        return self.sublayer_connection(self_att, network_layer, dropout)

    def decoder_module(self, inputs, enc_outputs, num_units, heads, dropout):
        masked_multi_head_att = self.multi_head_attention(inputs, inputs, inputs, heads, masked=True)
        masked_self_att = self.sublayer_connection(inputs, masked_multi_head_att, dropout)
        self_att = self.sublayer_connection(masked_self_att, self.multi_head_attention(masked_self_att, enc_outputs, enc_outputs, heads), dropout)
        ## conv_1d_layer or fnn 
        network_layer = self.feed_forward(self_att, num_units, dropout)
        return self.sublayer_connection(self_att, network_layer, dropout)
        
    def encoder(self, inputs, num_units, heads, num_layers, dropout):
        outputs = inputs 
        for _ in range(num_layers):
            outputs = self.encoder_module(outputs, num_units, heads, dropout)
        return outputs

    def decoder(self, inputs, enc_outputs, num_units, heads, num_layers, dropout):
        outputs = inputs 
        for _ in range(num_layers):
            outputs = self.decoder_module(outputs, enc_outputs, num_units, heads, dropout)
        return outputs

    def __init__(self, uc_data, batch_size, heads_size, layer_size, n_hidden, n_class):
        ## input params
        self.targets = tf.placeholder(tf.int64, [None, None], name='target') # (batch, step)
        self.enc_inputs = tf.placeholder(tf.int64, [None, None], name='enc_inputs') # (batch, step)
        self.dec_inputs = tf.placeholder(tf.int64, [None, None], name='dec_inputs') # (batch, step)
        self.out_keep_prob = tf.placeholder(tf.float32, name="out_keep_prob") # dropout
        self.target_seq_length = tf.placeholder(tf.int64, name="self.target_seq_length") 
        ## 
        self.positional_inputs = tf.tile(tf.range(0, uc_data.max_sequence_length), [batch_size])
        self.positional_inputs = tf.reshape(self.positional_inputs, [batch_size, uc_data.max_sequence_length])
        ## local embedding
        self.enc_embeddings = tf.Variable(tf.random_normal([uc_data.tot_dic_len, n_hidden]))
        self.dec_embeddings = tf.Variable(tf.random_normal([uc_data.tot_dic_len, n_hidden]))
        self.pos_endcodings = self.positional_encoding(n_hidden, uc_data.max_sequence_length)
        self.pos_endcodings.trainable = False

        self.position_encoded = tf.nn.embedding_lookup(self.pos_endcodings, self.positional_inputs)
        ## encoder 
        with tf.variable_scope('encode'):
            ## input embedding 
            self.enc_input_embeddings = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs) 
            # [?,?,256], [50,1047,128].
            self.enc_input_pos = self.enc_input_embeddings + self.position_encoded
            self.enc_outputs = self.encoder(self.enc_input_pos, [n_hidden * 4, n_hidden], heads_size, layer_size, self.out_keep_prob)
        ## decoder 
        with tf.variable_scope('decode'):
            ## input embedding 
            self.dec_input_embeddings = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs) # (batch size, sequence length , hidden size)
            self.dec_input_pos = self.dec_input_embeddings + self.position_encoded # (50, 1047, 128)
            self.dec_outputs = self.decoder(self.dec_input_pos, self.enc_outputs, [n_hidden * 4, n_hidden], heads_size, layer_size, self.out_keep_prob) # (50, 1047, 128)
        self.logits = tf.layers.dense(self.dec_outputs, n_class, activation=None, reuse=tf.AUTO_REUSE, name='output_layer')
        ##
        self.t_mask = tf.sequence_mask(self.target_seq_length, uc_data.max_sequence_length)
        self.t_mask.set_shape([batch_size, uc_data.max_sequence_length])
        ## loss
        with tf.variable_scope("loss"):
            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets)
            self.losses = tf.boolean_mask(self.losses, self.t_mask) 
            self.loss = tf.reduce_mean(self.losses)
        ## accuracy
        with tf.variable_scope("accuracy"):
            self.predict = tf.argmax(self.logits, 2)
            #self.predict_mask = self.prediction * tf.to_int64(self.t_mask)
            # (50, 1047) and (50, 22)
            self.predict_mask = tf.boolean_mask(self.predict, self.t_mask)
            self.targets_mask = tf.boolean_mask(self.targets, self.t_mask) 
            self.correct_pred = tf.equal(self.predict_mask, self.targets_mask)
            #self.correct_pred = tf.equal(self.predict, self.targets)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, "float"), name="accuracy")


