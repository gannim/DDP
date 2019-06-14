# -*- coding: utf-8 -*-
# original code : https://github.com/changwookjun/Transformer
# paper : https://arxiv.org/abs/1706.03762

import tensorflow as tf
import sys

import numpy as np

class Transformer(object):
    def __init__(self, uc_data, batch_size, heads_size, layer_size, n_hidden, n_class):
        ## input params
        self.enc_inputs = tf.placeholder(tf.int64, [None, None], name='enc_inputs') 
        self.dec_inputs = tf.placeholder(tf.int64, [None, None], name='dec_inputs')
        self.out_keep_prob = tf.placeholder(tf.float32, name="out_keep_prob") 
        
        ## answer params
        self.answers = tf.placeholder(tf.int64, [None, None], name='answers')
        self.answer_sequence_lengths = tf.placeholder(tf.int64, name="answer_sequence_lengths") 

        ## positional embeddings
        self.positional_inputs = tf.tile(tf.range(0, uc_data.max_sequence_length), [batch_size])
        self.positional_inputs = tf.reshape(self.positional_inputs, [batch_size, uc_data.max_sequence_length])
        self.pos_encodings = self.positional_encoding(n_hidden, uc_data.max_sequence_length)
        self.pos_encodings.trainable = False

        ## local embedding
        self.enc_embeddings = tf.Variable(tf.random_normal([uc_data.tot_dic_len, n_hidden]))
        self.dec_embeddings = tf.Variable(tf.random_normal([uc_data.tot_dic_len, n_hidden]))
        
        ## input - output embedding 
        self.enc_input_embeddings = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs) 
        self.dec_input_embeddings = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs) 
        
        ## positional encoding 
        self.position_encoded = tf.nn.embedding_lookup(self.pos_encodings, self.positional_inputs)
        self.enc_input_pos = self.enc_input_embeddings + self.position_encoded
        self.dec_input_pos = self.dec_input_embeddings + self.position_encoded 
        ## enc_input_pos, dec_input_pos 에 dropout 을 먹이기도 하는듯 ?

        ## encoding layer 
        with tf.variable_scope('encode'):
            self.enc_outputs = self.encoder(self.enc_input_pos, [n_hidden * 4, n_hidden], heads_size, layer_size, self.out_keep_prob)

        ## decoding layer 
        with tf.variable_scope('decode'):
            self.dec_outputs = self.decoder(self.dec_input_pos, self.enc_outputs, [n_hidden * 4, n_hidden], heads_size, layer_size, self.out_keep_prob)

        ## output layer
        self.logits = tf.layers.dense(self.dec_outputs, n_class, activation=None, reuse=tf.AUTO_REUSE, name='output_layer')
        
        ## mask
        self.t_mask = tf.sequence_mask(self.answer_sequence_lengths, uc_data.max_sequence_length)
        self.t_mask.set_shape([batch_size, uc_data.max_sequence_length])

        ## loss
        with tf.variable_scope("loss"):
            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.answers)
            self.losses = tf.boolean_mask(self.losses, self.t_mask) 
            self.loss = tf.reduce_mean(self.losses)

        ## accuracy
        with tf.variable_scope("accuracy"):
            self.predict = tf.argmax(self.logits, 2)
            self.predict_mask = tf.boolean_mask(self.predict, self.t_mask)
            self.targets_mask = tf.boolean_mask(self.answers, self.t_mask) 
            self.correct_pred = tf.equal(self.predict_mask, self.targets_mask)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, "float"), name="accuracy")

    def encoder(self, outputs, num_units, heads, num_layers, dropout):
        for _ in range(num_layers):
            outputs = self.encoder_module(outputs, num_units, heads, dropout)
        return outputs

    def decoder(self, outputs, enc_outputs, num_units, heads, num_layers, dropout):
        for _ in range(num_layers):
            outputs = self.decoder_module(outputs, enc_outputs, num_units, heads, dropout)
        return outputs

    def encoder_module(self, inputs, num_units, heads, dropout):
        att = self.multi_head_attention(inputs, inputs, inputs, heads)
        con = self.sublayer_connection(inputs, att, dropout)
        ##
        ffn = self.feed_forward(con, num_units, dropout)
        return self.sublayer_connection(con, ffn, dropout)

    def decoder_module(self, inputs, enc_outputs, num_units, heads, dropout):
        masked_att = self.multi_head_attention(inputs, inputs, inputs, heads, masked=True)
        masked_con = self.sublayer_connection(inputs, masked_att, dropout)
        ## 
        att = self.multi_head_attention(masked_con, enc_outputs, enc_outputs, heads)
        con = self.sublayer_connection(masked_con, att, dropout)
        ## 
        ffn = self.feed_forward(con, num_units, dropout)
        return self.sublayer_connection(con, ffn, dropout)

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
        
    def sublayer_connection(self, inputs, sublayer, dropout):
        # LayerNorm(x + Sublayer(x))
        #return tf.layers.dropout(self.layer_norm(inputs + sublayer), dropout)
        # Residual connection = input + sublayer 
        return self.layer_norm(inputs + sublayer)

    @staticmethod
    def layer_norm(inputs, epsilon=1e-8): #epsilon=1e-6):
        feature_shape = inputs.get_shape()[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(feature_shape), trainable=False)
        gamma = tf.Variable(tf.ones(feature_shape), trainable=False)
        #normalized = (inputs - mean) / ((variance + epsilon) ** 0.5) 
        normalized = (inputs - mean) / (variance + epsilon)
        return gamma * normalized + beta

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

    @staticmethod
    def feed_forward(inputs, num_units, dropout):
        # Position-wise Feed-Forward Networks
        # FFN(x) = max(0, xW1 + b1)W2 + b2 
        with tf.variable_scope("feed_forward", reuse=tf.AUTO_REUSE):
            # relu ==  max(0, x)
            # max(0, xW1 + b1)
            ## inner layer
            outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
            #outputs = tf.layers.dropout(outputs, dropout)
            # outer layer 
            return tf.layers.dense(outputs, num_units[1])

    @staticmethod
    def positional_encoding(dims, sentence_length, dtype=tf.float32):
        ary = []
        for i in range(dims):
            for pos in range(sentence_length):
                ary.append(pos/np.power(10000, 2*i/dims))
        encoded_vec = np.array(ary) 
        # PE(pos,2i) 
        encoded_vec[::2] = np.sin(encoded_vec[::2]) 
        # PE(pos,2i+1)
        encoded_vec[1::2] = np.cos(encoded_vec[1::2]) 
        return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dims]), dtype=dtype)
    
    @staticmethod 
    def noam_scheme(init_lr, gstep, warmup_step=4000):
        step = tf.cast(gstep + 1, dtype=tf.float32)
        return init_lr * warmup_steps ** 0.5 * tf.minimum(step *warmup_steps ** -1.5, step ** -0.5)

