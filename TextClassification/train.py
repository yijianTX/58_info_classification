#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, time
import tensorflow as tf
import numpy as np
import cPickle as pickle
from conv_neural_network import CNN
from data_pre_treatment import Pretreatment

log_file = open("classification.log", "w")

tf.set_random_seed(1000)

# --------- parameters --------- #
# CNN paras.
tf.flags.DEFINE_integer(flag_name = "sentence_max_length", default_value = 100, docstring = "maximum sentence cut off length.")
tf.flags.DEFINE_integer("number_classes", 43, "number of classes.")
tf.flags.DEFINE_float("l2_lmbda", 10.0, "L2 regularization lambda.")
tf.flags.DEFINE_float("dropout_prob", 0.5, "dropout keep probability.")
tf.flags.DEFINE_integer("embedding_size", 100, "word embedding size.")
tf.flags.DEFINE_string("filter_size_str", "2,3,4,5,6,7", "filter size string, using comma to seperate.")
tf.flags.DEFINE_integer("number_filters", 20, "number of filters for convolution layer.")
# Training paras.
tf.flags.DEFINE_integer("train_batch_size", 100, "number of batch size for training.")
tf.flags.DEFINE_integer("num_epoch", 20, "number of epochs for training.")
tf.flags.DEFINE_float("eta", 0.01, "learning rate for training.")
tf.flags.DEFINE_integer("checkpoint", 2000, "save model every these steps.")
tf.flags.DEFINE_integer("evaluate_point", 500, "evaluate the model after every these steps.")

tf.flags.FLAGS._parse_flags()
for key, value in tf.flags.FLAGS.__flags.items():
    print >> log_file,  "%s : %s" %(key, value)
FLAGS = tf.flags.FLAGS

# directories setting.
cur_time = time.strftime("%Y_%m_%d:%H_%M_%S", time.localtime())
out_dir = os.path.abspath(os.path.join(os.path.curdir, "result", cur_time))
train_summary_dir = os.path.join(out_dir, "summaries", "train")
validate_summary_dir = os.path.join(out_dir, "summaries", "validate")
test_summary_dir = os.path.join(out_dir, "summaries", "test")
checkpoint_dir = os.path.join(out_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

print >> log_file, "loading data begins..."
t1 = time.time()
pretreatment = Pretreatment(
        #data_dir_path = "/home/work/changjingya/traindata",
        data_dir_path = "/opt/Yujinhuang/info_classification/data", 
        dict_dir_path = "/opt/Yujinhuang/info_classification/dict", 
        sentence_max_length = FLAGS.sentence_max_length,
        use_segment = False
        )
pretreatment.load_data()
pretreatment.map_vocabs()
pretreatment.generate_input()
x, y, d_vocab, balanced_data_size = pretreatment.x, pretreatment.y, pretreatment.d_vocab, pretreatment.max_line
data_size = len(x)
indices = np.random.permutation(np.arange(data_size))

# shuffle data
x = list(np.array(x)[indices])
y = list(np.array(y)[indices])

# serialize datas
# f_dump = open("/opt/Yujinhuang/info_classification/data.bin", "wb")
# pickle.dump(x, f_dump)
# pickle.dump(y, f_dump)
# f_dump.close()
# f_load = open("/opt/Yujinhuang/info_classification/data.bin", "rb")
# x = pickle.load(f_load)
# y = pickle.load(f_load)
# f_load.close()
# data_size = len(y)

#serialize vocabs
# f_dump = open("/opt/Yujinhuang/info_classification/vocab.bin", "wb")
# pickle.dump(pretreatment.d_vocab, f_dump)
# f_dump.close()
# f_load = open("/opt/Yujinhuang/info_classification/vocab.bin", "rb")
# d_vocab = pickle.load(f_load)
# f_load.close()

vocab_size = len(d_vocab)
t2 = time.time()
print >> log_file, "loading data finished..."
print >> log_file, "loading data takes %.2f s" %(t2-t1)
print >> log_file, "vocabulary size : %d" %vocab_size
print >> log_file, "each balanced data size : %d" %balanced_data_size
#train_rate, validate_rate, test_rate = 0.8, 0.1, 0.1
#train_cut = int(data_size * train_rate)
#validate_cut = int(data_size * (train_rate + validate_rate))

x_train, x_validate, x_test = x[:-20000], x[-20000 : -10000], x[-10000:]
y_train, y_validate, y_test = y[:-20000], y[-20000 : -10000], y[-10000:]
#x_train, x_validate, x_test = x[:train_cut], x[train_cut : validate_cut], x[validate_cut:]
#y_train, y_validate, y_test = y[:train_cut], y[train_cut : validate_cut], y[validate_cut:]

print >> log_file, "training data size : %d" %len(x_train)
print >> log_file, "validate data size : %d" %len(x_validate)
print >> log_file, "test data size : %d" %len(x_test)

print >> log_file, "training model begins..."
with tf.Session() as sess:
    cnn = CNN(
        sentence_max_length = FLAGS.sentence_max_length, 
        number_classes = FLAGS.number_classes,
        l2_lmbda = FLAGS.l2_lmbda,
        dropout_prob = FLAGS.dropout_prob,
        vocab_size = vocab_size,
        embedding_size = FLAGS.embedding_size,
        filter_size_list = map(int, FLAGS.filter_size_str.split(",")),
        number_filters = FLAGS.number_filters
        )

    # define training operation.
    step_count = tf.Variable(initial_value = 0, trainable = False, name = "step_count")
    eta = tf.train.exponential_decay(learning_rate = FLAGS.eta, global_step = step_count, decay_steps = 100, decay_rate = 0.98)
    optimizer = tf.train.AdamOptimizer(learning_rate = eta)
    grads_and_vars_list = optimizer.compute_gradients(loss = cnn.loss)
    train_operation = optimizer.apply_gradients(grads_and_vars = grads_and_vars_list, global_step = step_count)

    # generate relative summaries for the use in TensorBoard.
    grad_summaries = []
    for g, v in grads_and_vars_list:
        if g is not None:
            grad_hist_summary = tf.summary.histogram(name = "%s_gradient_histogram" %v.name, values = g)
            grad_sparsity_summary = tf.summary.scalar(name = "%s_gradient_sparsity" %v.name, tensor = tf.nn.zero_fraction(value = g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(grad_sparsity_summary)
    grad_summaries_merged = tf.summary.merge(inputs = grad_summaries)
    loss_summary = tf.summary.scalar(name = "loss", tensor = cnn.loss)
    accuracy_summary = tf.summary.scalar(name = "accuracy", tensor = cnn.accuracy)

    # training summary
    train_sum_op = tf.summary.merge(inputs = [grad_summaries_merged, loss_summary, accuracy_summary])
    train_sum_writer = tf.summary.FileWriter(logdir = train_summary_dir, graph = sess.graph)

    # validate summary
    validate_sum_op = tf.summary.merge(inputs = [loss_summary, accuracy_summary])
    validate_sum_writer = tf.summary.FileWriter(logdir = validate_summary_dir, graph = sess.graph)

    # test summary
    test_sum_op = tf.summary.merge(inputs = [loss_summary, accuracy_summary])
    test_sum_writer = tf.summary.FileWriter(logdir = test_summary_dir, graph = sess.graph)

    def generate_batch_data(x, y, batch_size = FLAGS.train_batch_size):
        data = np.array(zip(x, y))
        num_batches = int(data_size / batch_size) + 1 
        for epoch in range(FLAGS.num_epoch):
            # shuffle data
            data_shuffled = np.random.permutation(data)
            for batch in range(num_batches):
                begin_idx = batch * batch_size
                end_idx = min(data_size, (batch + 1) * batch_size)
                yield data_shuffled[begin_idx : end_idx]

    def train_step(x_batch, y_batch):
        #print x_batch.shape(), y_batch.shape()
        # feed all the placeholders in cnn
        feed_dict = {
            cnn.input_x : x_batch,
            cnn.input_y : y_batch
        }
        _, step, train_summaries, loss, accuracy = sess.run(fetches = [train_operation, step_count, train_sum_op, cnn.loss, cnn.accuracy], feed_dict = feed_dict)
        print >> log_file, "training : step_count = %s, loss = %s, accuracy = %s" %(step, loss, accuracy)
        train_sum_writer.add_summary(summary = train_summaries, global_step = step_count.eval())

    def validate_step(x_batch, y_batch):
        feed_dict = {
            cnn.input_x : x_batch,
            cnn.input_y : y_batch
        }
        step, validate_summaries, loss, accuracy = sess.run(fetches = [step_count, validate_sum_op, cnn.loss, cnn.accuracy], feed_dict = feed_dict)
        print >> log_file, "validation : step_count = %s, loss = %s, accuracy = %s" %(step, loss, accuracy)
        validate_sum_writer.add_summary(summary = validate_summaries, global_step = step_count.eval())

    def test_step(x_batch, y_batch):
        feed_dict = {
            cnn.input_x : x_batch,
            cnn.input_y : y_batch
        }
        step, test_summaries, loss, accuracy = sess.run(fetches = [step_count, test_sum_op, cnn.loss, cnn.accuracy], feed_dict = feed_dict)
        print >> log_file, "test : step_count = %s, loss = %s, accuracy = %s" %(step, loss, accuracy)
        test_sum_writer.add_summary(summary = test_summaries, global_step = step_count.eval())

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_list = tf.global_variables())
    # data generators
    train_batches = generate_batch_data(x_train, y_train)
    for batch_data in train_batches:
        if not len(batch_data):
            continue
        x_batch, y_batch = zip(*batch_data)
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess = sess, global_step_tensor = step_count)
        if not current_step % FLAGS.evaluate_point:
            print >> log_file, "************ evaluation point ： %s ************" %str(current_step)
            validate_step(x_validate, y_validate)
        if not current_step % FLAGS.checkpoint:
            print >> log_file, "************ checkpoint ： %s ************" %str(current_step)
            _ = saver.save(sess = sess, save_path = checkpoint_dir, global_step = step_count)
    test_step(x_test, y_test)
    saver.save(sess = sess, save_path = checkpoint_dir, global_step = step_count)

