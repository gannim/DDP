#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import sys
import os 
import time

reload(sys)
sys.setdefaultencoding('utf-8')

from data_helper import dataHelper
from seq2seq import SEQ2SEQ

EVAL_EVERY = 100
NUM_CHECKPOINTS = 5
CHECKPOINT_EVERY = 1000

filename = 'data.txt'
uc_data = dataHelper(filename)
cell_type = 'bi-lstm'
batch_size = 50
n_hidden = 128
epochs = 100
n_class = len(uc_data.tot_word_idx_dic)
##
learning_rate = 0.001
keep_prob = 0.5
seq2seq = SEQ2SEQ(uc_data, cell_type, batch_size, n_hidden, n_class, True)
sources_train, sources_dev, outputs_train, outputs_dev, targets_train, targets_dev = uc_data.get_suffled_data()

train_set = np.array([(x, outputs_train[idx], targets_train[idx]) for idx, x in enumerate(sources_train)])
dev_set = np.array([(x, outputs_dev[idx], targets_dev[idx]) for idx, x in enumerate(sources_dev)])
    
def transiteration(sess, input_word):
    x_bat = uc_data.get_input_idxs(input_word) # [[44 45 42]] 
    x_len = np.array([len(x_bat[0])]) # [[3]]
    y_bat = uc_data.get_input_idxs(uc_data.START_SYMBOL) #[[1]] 
    result = sess.run(seq2seq.inf_result, feed_dict={seq2seq.enc_inputs:x_bat, seq2seq.inf_dec_inputs:y_bat, seq2seq.x_sequence_length:x_len, seq2seq.out_keep_prob:1.0})
    translated = uc_data.get_translated_str(result)
    return translated

def get_tfconfig():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    return config 

def get_summary(model, out_dir, graph):
    loss_summary = tf.summary.scalar("loss", model.loss)
    acc_summary = tf.summary.scalar("accuracy", model.accuracy)

    train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    train_summary_merge = tf.summary.merge_all()
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, graph)
    
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, graph)

    return train_summary_writer, train_summary_merge, dev_summary_writer, dev_summary_op

def get_checkpoint_prefix(out_dir):
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_prefix

def run_dev(dev_batchs, model, dev_summery_writer, dev_summary_op, sess, dev_summary_writer, cur_step):
    avg_dev_loss, avg_dev_acc = 0, 0

    for dev_ep, dev_batch in enumerate(dev_batchs):
        x_bat, y_bat, t_bat = zip(*dev_batch)
        nx_bat, x_seq_len = uc_data.pad(x_bat, batch_size, uc_data.max_inputs_seq_length)
        ny_bat, y_seq_len = uc_data.pad(y_bat, batch_size, uc_data.max_outputs_seq_length)
        nt_bat, t_seq_len = uc_data.pad(t_bat, batch_size, uc_data.max_targets_seq_length)

        feed_opt = {
            'loss': model.loss,
            'accuracy': model.accuracy,
            'predict': model.predict,
            'dev_summary_op': dev_summary_op,
        }
        feed = {
            model.enc_inputs: nx_bat, 
            model.dec_inputs: ny_bat, 
            model.targets: nt_bat, 
            model.out_keep_prob:1.0, 
            model.x_sequence_length:x_seq_len, 
            model.y_sequence_length:y_seq_len, 
            model.t_sequence_length:t_seq_len
        }
        results = sess.run(feed_opt, feed_dict=feed)

        avg_dev_loss += results.get('loss')
        avg_dev_acc += results.get('accuracy')
        if dev_ep == 0:
            input_word = uc_data.get_translated_str(x_bat[0])
            print('inferance {} -> {}'.format(input_word, transiteration(sess, input_word)))
            print('devset    {} -> {}'.format(input_word, uc_data.get_translated_str(results.get('predict')[0])))
    blen = dev_ep+1
    dev_summary_writer.add_summary(results.get('dev_summary_op'), cur_step)
    print("\nEvaluation : dev loss = %.6f / dev acc = %.6f" %(avg_dev_loss/blen, avg_dev_acc/blen))

def run_train():
    config = get_tfconfig()
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    with tf.Session(config=config) as sess:
        gstep = tf.Variable(0, name="gstep", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads_vars = optimizer.compute_gradients(seq2seq.loss)
        train_op = optimizer.apply_gradients(grads_vars, global_step=gstep)

        ## define summary
        train_summary_writer, train_summary_merge, dev_summary_writer, dev_summary_op = get_summary(seq2seq, out_dir, sess.graph) 

        ## define checkpoint dir
        checkpoint_prefix = get_checkpoint_prefix(out_dir)
        uc_data.save_vocab(os.path.join(out_dir, "vocab"), uc_data.tot_dic_len, uc_data.tot_word_idx_dic)
        
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=NUM_CHECKPOINTS)
        sess.run(tf.global_variables_initializer())

        ## train 
        batchs = uc_data.batch_iter(train_set, batch_size, epochs)
        avg_loss, avg_acc = 0, 0
        for epoch, batch in enumerate(batchs):
            x_bat, y_bat, t_bat = zip(*batch)

            nx_bat, x_seq_len = uc_data.pad(x_bat, batch_size, uc_data.max_inputs_seq_length)
            ny_bat, y_seq_len = uc_data.pad(y_bat, batch_size, uc_data.max_outputs_seq_length)
            nt_bat, t_seq_len = uc_data.pad(t_bat, batch_size, uc_data.max_targets_seq_length)

            feed_opt = {
                'gstep': gstep,
                'loss':seq2seq.loss,
                'accuracy':seq2seq.accuracy,
                'train_summary_merge': train_summary_merge,
                'train_op': train_op,
            }
            feed = {
                seq2seq.enc_inputs:nx_bat, 
                seq2seq.dec_inputs:ny_bat, 
                seq2seq.targets: nt_bat, 
                seq2seq.out_keep_prob:keep_prob, 
                seq2seq.x_sequence_length:x_seq_len, 
                seq2seq.y_sequence_length:y_seq_len, 
                seq2seq.t_sequence_length:t_seq_len
            }
            results = sess.run(feed_opt, feed_dict=feed)

            avg_loss += results.get('loss')
            avg_acc  += results.get('accuracy')
            cur_step = tf.train.global_step(sess, gstep)
            train_summary_writer.add_summary(results.get('train_summary_merge'), results.get('gstep'))
            
            if cur_step % EVAL_EVERY == 0:
                ## train avg loss , avg acc
                print('Epoch: %04d loss = %.6f / avg acc = %.6f' % (cur_step, avg_loss/EVAL_EVERY, avg_acc/EVAL_EVERY))
                avg_loss, avg_acc = 0, 0

                dev_batchs = uc_data.batch_iter(dev_set, batch_size, 1)
                run_dev(dev_batchs, seq2seq, dev_summary_writer, dev_summary_op, sess, dev_summary_writer, cur_step)

            if cur_step % CHECKPOINT_EVERY == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=cur_step)
                print("Saved model checkpoint to {}\n".format(path) )
        #학습 다하고 최종!
        print("final inferance")
        input_word = 'apple'
        translated = transiteration(sess, input_word)
        print('{} -> {}'.format(input_word, translated))
run_train()
