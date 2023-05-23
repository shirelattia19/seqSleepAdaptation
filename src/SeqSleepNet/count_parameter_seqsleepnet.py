import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0,-1"
import numpy as np
import tensorflow as tf

import shutil, sys
from datetime import datetime
import h5py

from seqsleepnet import SeqSleepNet
from seqsleepnet_config import Config


from scipy.io import loadmat, savemat
from tensorflow.python import pywrap_tensorflow

import time

# Parameters
# ==================================================

# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


tf.app.flags.DEFINE_float("dropout_keep_prob_rnn", 0.75, "Dropout keep probability (default: 0.75)")
tf.app.flags.DEFINE_integer("seq_len", 20, "Sequence length (default: 10)")
tf.app.flags.DEFINE_integer("nfilter", 32, "Sequence length (default: 32)")
tf.app.flags.DEFINE_integer("nhidden1", 64, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("attention_size1", 64, "Sequence length (default: 64)")
tf.app.flags.DEFINE_integer("nhidden2", 64, "Sequence length (default: 20)")

tf.app.flags.DEFINE_integer("nchannel", 1, "number of channel used (default: 1)")


FLAGS = tf.app.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()): # python3
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparatopn
# ==================================================

config = Config()
config.dropout_keep_prob_rnn = FLAGS.dropout_keep_prob_rnn
config.epoch_seq_len = FLAGS.seq_len
config.epoch_step = FLAGS.seq_len
config.nfilter = FLAGS.nfilter
config.nhidden1 = FLAGS.nhidden1
config.nhidden2 = FLAGS.nhidden2
config.attention_size1 = FLAGS.attention_size1
config.nchannel = FLAGS.nchannel


with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=False)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        net = SeqSleepNet(config=config)


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(config.learning_rate)
            grads_and_vars = optimizer.compute_gradients(net.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            print(shape)
            print(len(shape))
            variable_parameters = 1
            for dim in shape:
                print(dim)
                variable_parameters *= dim.value
            print(variable_parameters)
            total_parameters += variable_parameters
        print(total_parameters)
