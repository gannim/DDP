# -*- coding: utf-8 -*-
# original code : https://github.com/changwookjun/Transformer
# paper : https://arxiv.org/abs/1706.03762

import tensorflow as tf
import sys

import numpy as np
from attention import Attention

class SuperCoder:
    def __init__(self):
        self.att = Attention()

    def sublayer_connection(self, inputs, sublayer, dropout):
        # LayerNorm(x + Sublayer(x))
        # Residual connection = input + sublayer 
        return tf.contrib.layers.layer_norm(inputs + sublayer, scope="layer_norm", reuse=tf.AUTO_REUSE)

    def feed_forward(self, inputs, num_units, dropout):
        # Position-wise Feed-Forward Networks
        # FFN(x) = max(0, xW1 + b1)W2 + b2 
        with tf.variable_scope("feed_forward", reuse=tf.AUTO_REUSE):
            # relu ==  max(0, x)
            # max(0, xW1 + b1)
            ## inner layer
            outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu, name="filter_layer")
            outputs = tf.layers.dropout(outputs, dropout)
            # outer layer 
            return tf.layers.dense(outputs, num_units[1])

class Encoder(SuperCoder):
    def module(self, inputs, num_units, heads, dropout, enc_seq_len):
        self.att_out = self.att.multi_head_attention(inputs, inputs, inputs, heads, enc_seq_len)
        self.con = self.sublayer_connection(inputs, self.att_out, dropout)
        ##
        self.ffn = self.feed_forward(self.con, num_units, dropout)
        return self.sublayer_connection(self.con, self.ffn, dropout)

class Decoder(SuperCoder):
    def module(self, inputs, enc_outputs, num_units, heads, dropout, enc_seq_len, dec_seq_len):
        self.masked_att = self.att.multi_head_attention(inputs, inputs, inputs, heads, dec_seq_len, masked=True)
        #with tf.variable_scope('multihead_attention/rep_q', reuse=True):
        #with tf.variable_scope('multihead_attention/Wo', reuse=True):
        #    w = tf.get_variable('kernel')
        #self.masked_att = tf.Print(self.masked_att, [w], " ----> dec rep_q [w]")
        #self.masked_att = tf.Print(self.masked_att, [w], " ----> dec Wo [w]")
        #self.masked_att = tf.Print(self.masked_att, [self.masked_att], ">> masked_att")
        self.masked_con = self.sublayer_connection(inputs, self.masked_att, dropout)
        ## 
        self.att_out = self.att.multi_head_attention(self.masked_con, enc_outputs, enc_outputs, heads, enc_seq_len)
        self.con = self.sublayer_connection(self.masked_con, self.att_out, dropout)
        ## 
        self.ffn = self.feed_forward(self.con, num_units, dropout)
        return self.sublayer_connection(self.con, self.ffn, dropout)

class Transformer(object):
    def __init__(self, uc_data, batch_size, heads_size, layer_size, n_hidden, n_class):
        ## 
        self.encoder = Encoder()
        self.decoder = Decoder()
        ## input params
        self.enc_inputs = tf.placeholder(tf.int64, [None, None], name='enc_inputs') 
        self.dec_inputs = tf.placeholder(tf.int64, [None, None], name='dec_inputs')
        self.enc_seq_len = tf.placeholder(tf.float32, name="enc_seq_len") 
        self.dec_seq_len = tf.placeholder(tf.float32, name="dec_seq_len") 
        self.out_keep_prob = tf.placeholder(tf.float32, name="out_keep_prob") 
        
        ## answer params
        self.answers = tf.placeholder(tf.int64, [None, None], name='answers')
        self.answer_sequence_lengths = tf.placeholder(tf.int64, name="answer_sequence_lengths") 

        ## positional embeddings
        self.positional_inputs = tf.range(0, uc_data.max_sequence_length)
        self.pos_encodings = self.positional_encoding(n_hidden, uc_data.max_sequence_length)
        self.pos_encodings.trainable = False

        ## positional encoding 
        with tf.variable_scope('positional_encoding', reuse=tf.AUTO_REUSE):
            self.pos_input_embeddings = tf.nn.embedding_lookup(self.pos_encodings, self.positional_inputs, name="pos_input_embeddings")

        with tf.variable_scope('encoder_input', reuse=tf.AUTO_REUSE):
            self.enc_embeddings = tf.Variable(tf.random_normal([uc_data.tot_dic_len, n_hidden]), name="enc_embeddings")
            self.enc_input_embeddings = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs, name="enc_input_embeddings") 
            self.enc_input_pos = self.enc_input_embeddings + self.pos_input_embeddings

        with tf.variable_scope('decoder_input', reuse=tf.AUTO_REUSE):
            self.dec_embeddings = tf.Variable(tf.random_normal([uc_data.tot_dic_len, n_hidden]), name="dec_embeddings")
            self.dec_input_embeddings = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs, name="dec_input_embeddings") 
            self.dec_input_pos = self.dec_input_embeddings + self.pos_input_embeddings

        ## encoding layer 
        with tf.variable_scope('encode', reuse=tf.AUTO_REUSE):
            self.enc_outputs = self._encoder(self.enc_input_pos, [n_hidden * 4, n_hidden], heads_size, layer_size, self.out_keep_prob, self.enc_seq_len)

        ## decoding layer 
        with tf.variable_scope('decode', reuse=tf.AUTO_REUSE):
            self.dec_outputs = self._decoder(self.dec_input_pos, self.enc_outputs, [n_hidden * 4, n_hidden], heads_size, layer_size, self.out_keep_prob, self.enc_seq_len, self.dec_seq_len)

        ## output layer
        self.logits = tf.layers.dense(self.dec_outputs, n_class, activation=None, reuse=tf.AUTO_REUSE, name='output_layer')
        
        ## predict, loss, accuracy
        self.predict, self.loss, self.accuracy = self.predict_loss_accuracy(self.logits, self.answers, batch_size, self.answer_sequence_lengths, uc_data.max_sequence_length)

        ## inference 
        self.inference(uc_data, n_hidden, heads_size, layer_size, n_class)

    def inference(self, uc_data, n_hidden, heads_size, layer_size, n_class):
        ## inputs 
        self.dec_step_inputs = tf.placeholder(tf.int64, [None, None], name='dec_step_inputs')
        self.end_symbol_idx = tf.convert_to_tensor(np.array([[2]]), dtype=tf.int64) # cond
        ## ouptut 
        self.output_tensor_t = tf.TensorArray(tf.int64, size = 0, dynamic_size=True, name="output") 

        def false():
            return False
        def true():
            return True

        def cond(i, pred, next_inputs, enc_outputs, ot, es):
            p = tf.reshape(pred, [])
            e = tf.reshape(es, [])
            return tf.case({tf.greater_equal(i, uc_data.max_targets_seq_length): false, tf.equal(p, e): false}, default=true)
            #return tf.case({tf.greater_equal(i, uc_data.max_targets_seq_length): false}, default=true)

        def body(i, cur_input, dec_input, enc_outputs, output_tensor_t, es):
            # dec_input shape [1, 1]
            with tf.variable_scope('decoder_input', reuse=tf.AUTO_REUSE):
                # dec_embedding print 결과 잘 쓰고 있음
                params = tf.range(0, i+1)
                self.dec_step_input_embeddings = tf.nn.embedding_lookup(self.dec_embeddings, dec_input, name="dec_step_input_embeddings")
                pe_i = tf.nn.embedding_lookup(self.pos_encodings, params, name="pos_input_embeddings")
                #pe_i = tf.gather_nd(self.pos_input_embeddings, [params]) # step, hidden

                # [2 512][1 2 512] 
                #self.dec_step_input_embeddings = tf.Print(self.dec_step_input_embeddings, [tf.shape(pe_i), tf.shape(self.dec_step_input_embeddings)], ">> shape")
                self.dec_step_input_pos = self.dec_step_input_embeddings + pe_i # batch, seq, hidden
                self.dec_step_input_pos = tf.reshape(self.dec_step_input_pos, [1, i+1, n_hidden], name="dec_step_input_pos")
            ## decoding layer 
            with tf.variable_scope('decode', reuse=tf.AUTO_REUSE):
                self.dec_step_output = self._decoder(self.dec_step_input_pos, enc_outputs, [n_hidden * 4, n_hidden], heads_size, layer_size, self.out_keep_prob, self.enc_seq_len, i+1)
            step_logits = tf.layers.dense(self.dec_step_output, n_class, activation=None, reuse=tf.AUTO_REUSE, name='output_layer')
            step_predict = tf.argmax(step_logits, 2)
            step_predict = tf.reshape(tf.gather_nd(step_predict, [0, i]), [1, -1])
            output_tensor_t = output_tensor_t.write( i, step_predict )
            next_inputs = tf.concat([dec_input, step_predict], -1)
            i = tf.Print(i, [i, dec_input, step_predict, tf.shape(step_predict), tf.shape(next_inputs)], "i, dec_input, pred")
            return i+1, step_predict, next_inputs, enc_outputs, output_tensor_t, es
        _, _, _, _, self.output_tensor_t, _ = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[tf.constant(0), self.dec_step_inputs, self.dec_step_inputs, self.enc_outputs, self.output_tensor_t, self.end_symbol_idx],
            name="inf_decoder")
        self.inf_result = self.output_tensor_t.stack()
        self.inf_result = tf.reshape( self.inf_result, [-1] , name='inf_result') 

    def _encoder(self, outputs, num_units, heads, num_layers, dropout, enc_seq_len):
        for _ in range(num_layers):
            outputs = self.encoder.module(outputs, num_units, heads, dropout, enc_seq_len)
        return outputs

    def _decoder(self, outputs, enc_outputs, num_units, heads, num_layers, dropout, enc_seq_len, dec_seq_len):
        for i in range(num_layers):
            outputs = self.decoder.module(outputs, enc_outputs, num_units, heads, dropout, enc_seq_len, dec_seq_len)
        return outputs
        
    def predict_loss_accuracy(self, logits, answers, batch_size, answer_sequence_length, max_sequence_length):
        ## mask
        with tf.variable_scope("mask"):
            self.t_mask = tf.sequence_mask(answer_sequence_length, max_sequence_length)
            self.t_mask.set_shape([batch_size, max_sequence_length])

        ## loss
        with tf.variable_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=answers)
            losses = tf.boolean_mask(losses, self.t_mask) 
            self.loss = tf.reduce_mean(losses)

        ## accuracy
        with tf.variable_scope("accuracy"):
            self.predict = tf.argmax(logits, 2)
            predict_mask = tf.boolean_mask(self.predict, self.t_mask)
            targets_mask = tf.boolean_mask(answers, self.t_mask) 
            correct_pred = tf.equal(predict_mask, targets_mask)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"), name="accuracy")
        return self.predict, self.loss, self.accuracy

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
    def noam_scheme(init_lr, gstep, warmup_steps=4000):
        step = tf.cast(gstep + 1, dtype=tf.float32)
        return init_lr * warmup_steps ** 0.5 * tf.minimum(step *warmup_steps ** -1.5, step ** -0.5)


