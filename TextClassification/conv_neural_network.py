#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

class CNN(object):
	"""
	define all the layers in the convolutional neural network.
	tensorflow version is r1.4.
	basiclly we use an embedding layer, then a convolutional layer, followed by a pooling layer, end with a softmax layer. 
	we can add more convolutional layers + pooling layers if model underfits.
	we use RELU as our activation functions to avoid gradient vanishing.
	we use gradient clipping to avoid gradient explosion.
	we use dropout and l2 regularization to avoid overfitting.
	"""

	def __init__(self, sentence_max_length, number_classes, l2_lmbda, dropout_prob, vocab_size, embedding_size, filter_size_list, number_filters):

		# input and output.
		self.input_x = tf.placeholder(dtype = tf.int32, shape = [None, sentence_max_length], name = "input_x")
		self.input_y = tf.placeholder(dtype = tf.float32, shape = [None, number_classes], name = "input_y")

		# L2 regularization loss param lambda and dropout probability.
		self.l2_lmbda = l2_lmbda
		self.dropout_prob = dropout_prob

		# embedding layer.
		with tf.name_scope(name = "embedding_layer"):
			W = tf.Variable(initial_value = tf.random_uniform(shape = [vocab_size, embedding_size], minval = -1.0, maxval = 1.0), name = "W")
			# tensor shape is : None * sentence_max_length * embedding_size.
			self.embedded_input = tf.nn.embedding_lookup(params = W, ids = self.input_x)
			# add one dimension to 4-D tensor
			self.embedded_input_expanded = tf.expand_dims(input = self.embedded_input, axis = -1)

		# first convolution layer and max pooling layer.
		first_pooled_output_list = []
		for filter_size in filter_size_list:
			with tf.name_scope(name = "1st_conv_layer_filter_size_%s" % filter_size):
				# see conv to know the meaning.
				filter_shape = [filter_size, embedding_size, 1, number_filters]
				# values more than 2 stddev from the mean are dropped.
				W = tf.Variable(initial_value = tf.truncated_normal(shape = filter_shape, stddev = 0.1), name = "W")
				b = tf.Variable(initial_value = tf.constant(value = 0.1, shape = [number_filters]), name = "b")

				# convolution operation.
				# 	input shape meaning : batch_size, in_height, in_width, in_channel -------- None * sentence_max_length * embedding_size * 1.
				# 	filter shape meaning : win_height, win_width, in_channel, out_channel --------- filter_size * embedding_size * 1 * number_filters .
				# 	strides : different strides for windows to move in 4 dimensions.
				# 	padding : ways to do the convolution, "valid" means the result tensor shape will be changed after convolution.
				# 	conv shape meaning : batch_siz0e, out_height, out_width, out_channel -------- None * ((sentence_max_length - filter_size) / stride + 1) * 1 * number_filters.
				conv = tf.nn.conv2d(input = self.embedded_input_expanded, filter = W, strides = [1, 1, 1, 1], padding = "VALID", name = "conv")

				# use relu to activate our results, relu has the same shape with features.
				relu = tf.nn.relu(features = tf.nn.bias_add(value = conv, bias = b), name = "relu")

				# use max pooling on the results.
				# 	ksize = the size of window for different dimensions.
				# 	pooled_res shape : None * 1 * 1 * number_filters.
				pooled_res = tf.nn.max_pool(value = relu, ksize = [1, sentence_max_length - filter_size + 1, 1, 1], strides = [1, 1, 1, 1], padding = "VALID", name = "pooling")
				first_pooled_output_list.append(pooled_res)

		# convert list of tensors into tensor using concat, shape : None * 1 * 1 * first_number_feature_map.
		first_number_feature_map = number_filters * len(filter_sizes)
		self.first_pool = tf.concat(values = first_pooled_output_list, axis = -1)
		self.first_pool_flat = tf.reshape(tensor = self.first_pool, shape = [-1, first_number_feature_map])

		# add dropout.
		with tf.name_scope(name = "dropout"):
			self.h_drop = tf.nn.dropout(x = self.first_pool_flat, keep_prob = self.dropout_prob)

		# output layer.
		regular_term = tf.constant(value = 0.0)
		with tf.name_scope(name = "output_layer"):
			# should use get_variable to do the initialization.
			W = tf.get_variable(name = "W", shape = [first_number_feature_map, number_classes], initializer = tf.contrib.layers.xavier_initializer())
			b = tf.Variable(initial_value = tf.constant(value = 0.1, shape = [number_classes]), name = "b")
			regular_term += tf.nn.l2_loss(t = W)
			# shape : None * number_classes.
			self.h_out = tf.nn.xw_plus_b(x = self.h_drop, weights = W, biases = b, name = "h_out")
			self.softmax_out = tf.nn.softmax(logits = self.h_out, name = "softmax_out")
			# shape : None.
			self.prediction = tf.argmax(input = self.softmax_out, axis = -1, name = "prediction")

		# cross-entropy loss.
		with tf.name_scope("loss"):
			loss = tf.nn.softmax_cross_entropy_with_logits(labels = self.input_y, logits = self.softmax_out)
			self.loss = tf.reduce_mean(input_tensor = loss) + self.l2_lmbda * regular_term

		# accuracy.
		with tf.name_scope("accuracy"):
			y = tf.argmax(input = self.input_y, axis = -1, name = "y")
			self.accuracy = tf.reduce_mean(input_tensor = tf.cast(x = y, dtype = tf.float32), name = "accuracy")

