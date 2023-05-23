import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0,-1"
import numpy as np
import tensorflow as tf

import shutil, sys
from datetime import datetime
import h5py

from seqsleepnet import SeqSleepNet
from seqsleepnet_config import Config

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from datagenerator_from_list_v2 import DataGenerator

from scipy.io import loadmat, savemat
from tensorflow.python import pywrap_tensorflow

import time

# Parameters
# ==================================================

# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# My Parameters
tf.app.flags.DEFINE_string("eeg_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eeg_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eeg_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("out_dir", "./output/", "Point to output directory")
tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")

tf.app.flags.DEFINE_float("dropout_keep_prob_rnn", 0.75, "Dropout keep probability (default: 0.75)")
tf.app.flags.DEFINE_integer("seq_len", 20, "Sequence length (default: 10)")
tf.app.flags.DEFINE_integer("nfilter", 32, "Sequence length (default: 32)")
tf.app.flags.DEFINE_integer("nhidden1", 64, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("attention_size1", 64, "Sequence length (default: 64)")
tf.app.flags.DEFINE_integer("nhidden2", 64, "Sequence length (default: 20)")

# 0: All, 1: softmax+SPB, 2: softmax+EPB, 3: softmax
tf.app.flags.DEFINE_integer("finetune_mode", 0, "Finetuning mode")
# pretrained model checkpoint, set to empty if training from scratch
tf.app.flags.DEFINE_string("pretrained_model", "", "Point to the pretrained model checkpoint")

#tf.app.flags.DEFINE_integer("evaluate_every", 1000, "Numer of training step to evaluate (default: 100)")

# flag for early stopping
tf.app.flags.DEFINE_boolean("early_stopping", False, "whether to apply early stopping (default: False)")
FLAGS = tf.app.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()): # python3
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparatopn
# ==================================================

# path where some output are stored
out_path = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
# path where checkpoint models are stored
checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))
if not os.path.isdir(os.path.abspath(out_path)): os.makedirs(os.path.abspath(out_path))
if not os.path.isdir(os.path.abspath(checkpoint_path)): os.makedirs(os.path.abspath(checkpoint_path))

config = Config()
config.dropout_keep_prob_rnn = FLAGS.dropout_keep_prob_rnn
config.epoch_seq_len = FLAGS.seq_len
config.epoch_step = FLAGS.seq_len
config.nfilter = FLAGS.nfilter
config.nhidden1 = FLAGS.nhidden1
config.nhidden2 = FLAGS.nhidden2
config.attention_size1 = FLAGS.attention_size1
#config.evaluate_every = FLAGS.evaluate_every

pretrained_model_dir = FLAGS.pretrained_model
finetune_mode = FLAGS.finetune_mode

eeg_active = ((FLAGS.eeg_train_data != "") and (FLAGS.eeg_test_data != ""))
eog_active = ((FLAGS.eog_train_data != "") and (FLAGS.eog_test_data != ""))
emg_active = ((FLAGS.emg_train_data != "") and (FLAGS.emg_test_data != ""))

# EEG data loader and generator
if (eeg_active):
    print("eeg active")
    # Initalize the data generator seperately for the training, validation, and test sets
    print('test' + os.path.abspath(FLAGS.eeg_train_data))
    eeg_train_gen = DataGenerator(os.path.abspath(FLAGS.eeg_train_data), data_shape=[config.frame_seq_len, config.ndim], seq_len=config.epoch_seq_len, shuffle = False)
    #eeg_test_gen = DataGenerator(os.path.abspath(FLAGS.eeg_test_data), data_shape=[config.frame_seq_len, config.ndim], seq_len=config.epoch_seq_len, shuffle = False)
    eeg_eval_gen = DataGenerator(os.path.abspath(FLAGS.eeg_eval_data), data_shape=[config.frame_seq_len, config.ndim], seq_len=config.epoch_seq_len, shuffle = False)

    # data normalization here
    X = eeg_train_gen.X
    X = np.reshape(X,(eeg_train_gen.data_size*eeg_train_gen.data_shape[0], eeg_train_gen.data_shape[1]))
    meanX = X.mean(axis=0)
    print('meanX')
    print(meanX)
    stdX = X.std(axis=0)
    print('stdX')
    print(stdX)
    X = (X - meanX) / stdX
    eeg_train_gen.X = np.reshape(X, (eeg_train_gen.data_size, eeg_train_gen.data_shape[0], eeg_train_gen.data_shape[1]))

    X = eeg_eval_gen.X
    X = np.reshape(X,(eeg_eval_gen.data_size*eeg_eval_gen.data_shape[0], eeg_eval_gen.data_shape[1]))
    X = (X - meanX) / stdX
    eeg_eval_gen.X = np.reshape(X, (eeg_eval_gen.data_size, eeg_eval_gen.data_shape[0], eeg_eval_gen.data_shape[1]))

    #X = eeg_test_gen.X
    #X = np.reshape(X,(eeg_test_gen.data_size*eeg_test_gen.data_shape[0], eeg_test_gen.data_shape[1]))
    #X = (X - meanX) / stdX
    #eeg_test_gen.X = np.reshape(X, (eeg_test_gen.data_size, eeg_test_gen.data_shape[0], eeg_test_gen.data_shape[1]))

# EOG data loader and generator
if (eog_active):
    print("eog active")
    # Initalize the data generator seperately for the training, validation, and test sets
    eog_train_gen = DataGenerator(os.path.abspath(FLAGS.eog_train_data), data_shape=[config.frame_seq_len, config.ndim], seq_len=config.epoch_seq_len, shuffle = False)
    #eog_test_gen = DataGenerator(os.path.abspath(FLAGS.eog_test_data), data_shape=[config.frame_seq_len, config.ndim], seq_len=config.epoch_seq_len, shuffle = False)
    eog_eval_gen = DataGenerator(os.path.abspath(FLAGS.eog_eval_data), data_shape=[config.frame_seq_len, config.ndim], seq_len=config.epoch_seq_len, shuffle = False)

    # data normalization here
    X = eog_train_gen.X
    X = np.reshape(X,(eog_train_gen.data_size*eog_train_gen.data_shape[0], eog_train_gen.data_shape[1]))
    meanX = X.mean(axis=0)
    stdX = X.std(axis=0)
    X = (X - meanX) / stdX
    eog_train_gen.X = np.reshape(X, (eog_train_gen.data_size, eog_train_gen.data_shape[0], eog_train_gen.data_shape[1]))

    X = eog_eval_gen.X
    X = np.reshape(X,(eog_eval_gen.data_size*eog_eval_gen.data_shape[0], eog_eval_gen.data_shape[1]))
    X = (X - meanX) / stdX
    eog_eval_gen.X = np.reshape(X, (eog_eval_gen.data_size, eog_eval_gen.data_shape[0], eog_eval_gen.data_shape[1]))

    #X = eog_test_gen.X
    #X = np.reshape(X,(eog_test_gen.data_size*eog_test_gen.data_shape[0], eog_test_gen.data_shape[1]))
    #X = (X - meanX) / stdX
    #eog_test_gen.X = np.reshape(X, (eog_test_gen.data_size, eog_test_gen.data_shape[0], eog_test_gen.data_shape[1]))

# EMG data loader and generator
if (emg_active):
    print("emg active")
    # Initalize the data generator seperately for the training, validation, and test sets
    emg_train_gen = DataGenerator(os.path.abspath(FLAGS.emg_train_data), data_shape=[config.frame_seq_len, config.ndim], seq_len=config.epoch_seq_len, shuffle = False)
    #emg_test_gen = DataGenerator(os.path.abspath(FLAGS.emg_test_data), data_shape=[config.frame_seq_len, config.ndim], seq_len=config.epoch_seq_len, shuffle = False)
    emg_eval_gen = DataGenerator(os.path.abspath(FLAGS.emg_eval_data), data_shape=[config.frame_seq_len, config.ndim], seq_len=config.epoch_seq_len, shuffle = False)

    # data normalization here
    X = emg_train_gen.X
    X = np.reshape(X,(emg_train_gen.data_size*emg_train_gen.data_shape[0], emg_train_gen.data_shape[1]))
    meanX = X.mean(axis=0)
    stdX = X.std(axis=0)
    X = (X - meanX) / stdX
    emg_train_gen.X = np.reshape(X, (emg_train_gen.data_size, emg_train_gen.data_shape[0], emg_train_gen.data_shape[1]))

    X = emg_eval_gen.X
    X = np.reshape(X,(emg_eval_gen.data_size*emg_eval_gen.data_shape[0], emg_eval_gen.data_shape[1]))
    X = (X - meanX) / stdX
    emg_eval_gen.X = np.reshape(X, (emg_eval_gen.data_size, emg_eval_gen.data_shape[0], emg_eval_gen.data_shape[1]))

    #X = emg_test_gen.X
    #X = np.reshape(X,(emg_test_gen.data_size*emg_test_gen.data_shape[0], emg_test_gen.data_shape[1]))
    #X = (X - meanX) / stdX
    #emg_test_gen.X = np.reshape(X, (emg_test_gen.data_size, emg_test_gen.data_shape[0], emg_test_gen.data_shape[1]))

# eeg always active
train_generator = eeg_train_gen
#test_generator = eeg_test_gen
eval_generator = eeg_eval_gen

# expand channel dimension if single channel EEG
if (not(eog_active) and not(emg_active)):
    train_generator.X = np.expand_dims(train_generator.X, axis=-1) # expand channel dimension
    train_generator.data_shape = train_generator.X.shape[1:]
    #test_generator.X = np.expand_dims(test_generator.X, axis=-1) # expand channel dimension
    #test_generator.data_shape = test_generator.X.shape[1:]
    eval_generator.X = np.expand_dims(eval_generator.X, axis=-1) # expand channel dimension
    eval_generator.data_shape = eval_generator.X.shape[1:]
    nchannel = 1
    print(train_generator.X.shape)

# stack in channel dimension if 2 channel EEG+EOG
if (eog_active and not(emg_active)):
    print(train_generator.X.shape)
    print(eog_train_gen.X.shape)
    train_generator.X = np.stack((train_generator.X, eog_train_gen.X), axis=-1) # merge and make new dimension
    train_generator.data_shape = train_generator.X.shape[1:]
    #test_generator.X = np.stack((test_generator.X, eog_test_gen.X), axis=-1) # merge and make new dimension
    #test_generator.data_shape = test_generator.X.shape[1:]
    eval_generator.X = np.stack((eval_generator.X, eog_eval_gen.X), axis=-1) # merge and make new dimension
    eval_generator.data_shape = eval_generator.X.shape[1:]
    nchannel = 2
    print(train_generator.X.shape)

# stack in channel dimension if 2 channel EEG+EOG+EMG
if (eog_active and emg_active):
    print(train_generator.X.shape)
    print(eog_train_gen.X.shape)
    print(emg_train_gen.X.shape)
    train_generator.X = np.stack((train_generator.X, eog_train_gen.X, emg_train_gen.X), axis=-1) # merge and make new dimension
    train_generator.data_shape = train_generator.X.shape[1:]
    #test_generator.X = np.stack((test_generator.X, eog_test_gen.X, emg_test_gen.X), axis=-1) # merge and make new dimension
    #test_generator.data_shape = test_generator.X.shape[1:]
    eval_generator.X = np.stack((eval_generator.X, eog_eval_gen.X, emg_eval_gen.X), axis=-1) # merge and make new dimension
    eval_generator.data_shape = eval_generator.X.shape[1:]
    nchannel = 3
    print(train_generator.X.shape)

config.nchannel = nchannel

del eeg_train_gen
#del eeg_test_gen
del eeg_eval_gen
if (eog_active):
    del eog_train_gen
    #del eog_test_gen
    del eog_eval_gen
if (emg_active):
    del emg_train_gen
    #del emg_test_gen
    del emg_eval_gen

# shuffle training data here
train_generator.shuffle_data()

train_batches_per_epoch = np.floor(len(train_generator.data_index) / config.batch_size).astype(np.uint32)
eval_batches_per_epoch = np.floor(len(eval_generator.data_index) / config.batch_size).astype(np.uint32)
#test_batches_per_epoch = np.floor(len(test_generator.data_index) / config.batch_size).astype(np.uint32)

#print("Train/Eval/Test set: {:d}/{:d}/{:d}".format(train_generator.data_size, eval_generator.data_size, test_generator.data_size))
#print("Train/Eval/Test batches per epoch: {:d}/{:d}/{:d}".format(train_batches_per_epoch, eval_batches_per_epoch, test_batches_per_epoch))

print("Train/Eval set: {:d}/{:d}".format(train_generator.data_size, eval_generator.data_size))
print("Train/Eval batches per epoch: {:d}/{:d}".format(train_batches_per_epoch, eval_batches_per_epoch))

# variable to keep track of best accuracy on validation set for model selection
best_acc = 0.0

# Training
# ==================================================
early_stop_count = 0

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45, allow_growth=False)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        net = SeqSleepNet(config=config)

        out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(config.learning_rate)

            if(pretrained_model_dir == ""):
                print('Scratch training ... ')
                grads_and_vars = optimizer.compute_gradients(net.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            else:
                finetune_vars = list()
                if(finetune_mode == 0): # All
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="filterbank-layer-eeg"))
                    if(config.nchannel > 1):
                        finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="filterbank-layer-eog"))
                    if(config.nchannel > 2):
                        finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="filterbank-layer-emg"))
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="frame_rnn_layer"))
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="frame_attention_layer"))
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="epoch_rnn_layer"))
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="output_layer"))
                elif(finetune_mode == 1): # softmax+SPB
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="epoch_rnn_layer"))
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="output_layer"))
                elif(finetune_mode == 2): # softmax+EPB
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="filterbank-layer-eeg"))
                    if(config.nchannel > 1):
                        finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="filterbank-layer-eog"))
                    if(config.nchannel > 2):
                        finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="filterbank-layer-emg"))
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="frame_rnn_layer"))
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="frame_attention_layer"))
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="output_layer"))
                elif(finetune_mode == 3): # softmax+EPB
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="filterbank-layer-eeg"))
                    if(config.nchannel > 1):
                        finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="filterbank-layer-eog"))
                    if(config.nchannel > 2):
                        finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="filterbank-layer-emg"))
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="output_layer"))
                elif(finetune_mode == 4): # softmax
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="output_layer"))
                else:
                    print('Inappropriate finetuning mode!')
                print('Finetuning ... ')
                print('FINETUNED VARIABLES')
                print(finetune_vars)
                grads_and_vars = optimizer.compute_gradients(net.loss, var_list=finetune_vars)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # initialize all variables
        sess.run(tf.global_variables_initializer())
        print("Model initialized")

        if(pretrained_model_dir != ""):
            variables = list()
            # only load variables of these scopes
            variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="filterbank-layer-eeg"))
            variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="frame_rnn_layer"))
            variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="frame_attention_layer"))
            variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="epoch_rnn_layer"))
            variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="output_layer"))

            print("RESTORE VARIABLES")
            #print(variables)
            for i, v in enumerate(variables):
                print(v.name[:-2])

            vars_in_checkpoint = tf.train.list_variables(pretrained_model_dir)
            print("IN-CHECK-POINT VARIABLES")
            #print(vars_in_checkpoint)
            vars_in_checkpoint_names = list()
            for i, v in enumerate(vars_in_checkpoint):
                print(v[0])
                vars_in_checkpoint_names.append(v[0])

            var_list_to_retstore = [v for v in variables if v.name[:-2] in vars_in_checkpoint_names]
            print("ACTUAL RESTORE VARIABLES")
            print(var_list_to_retstore)

            restorer = tf.train.Saver(var_list_to_retstore)
            restorer.restore(sess, pretrained_model_dir)
            print("Pretrained model loaded")


        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            frame_seq_len = np.ones(len(x_batch)*config.epoch_seq_len,dtype=int) * config.frame_seq_len
            epoch_seq_len = np.ones(len(x_batch),dtype=int) * config.epoch_seq_len
            feed_dict = {
              net.input_x: x_batch,
              net.input_y: y_batch,
              net.dropout_keep_prob_rnn: config.dropout_keep_prob_rnn,
              net.epoch_seq_len: epoch_seq_len,
              net.frame_seq_len: frame_seq_len,
              net.istraining: 1
            }
            _, step, output_loss, total_loss, accuracy = sess.run(
               [train_op, global_step, net.output_loss, net.loss, net.accuracy],
               feed_dict)
            return step, output_loss, total_loss, accuracy


        def dev_step(x_batch, y_batch):
            """
            A single evaluation step
            """
            frame_seq_len = np.ones(len(x_batch)*config.epoch_seq_len,dtype=int) * config.frame_seq_len
            epoch_seq_len = np.ones(len(x_batch),dtype=int) * config.epoch_seq_len
            feed_dict = {
                net.input_x: x_batch,
                net.input_y: y_batch,
                net.dropout_keep_prob_rnn: 1.0,
                net.epoch_seq_len: epoch_seq_len,
                net.frame_seq_len: frame_seq_len,
                net.istraining: 0
            }
            output_loss, total_loss, yhat = sess.run(
                   [net.output_loss, net.loss, net.predictions], feed_dict)
            return output_loss, total_loss, yhat

        def evaluate(gen, log_filename):
            # Validate the model on the entire data stored in gen variable
            output_loss =0
            total_loss = 0
            yhat = np.zeros([config.epoch_seq_len, len(gen.data_index)])
            # test with minibatch of 10x training minibatch to speed up
            num_batch_per_epoch = np.floor(len(gen.data_index) / (10*config.batch_size)).astype(np.uint32)
            test_step = 1
            while test_step < num_batch_per_epoch:
                x_batch, y_batch, label_batch_ = gen.next_batch(10*config.batch_size)
                output_loss_, total_loss_, yhat_ = dev_step(x_batch, y_batch)
                output_loss += output_loss_
                total_loss += total_loss_
                for n in range(config.epoch_seq_len):
                    yhat[n, (test_step-1)*10*config.batch_size : test_step*10*config.batch_size] = yhat_[n]
                test_step += 1
            if(gen.pointer < len(gen.data_index)):
                actual_len, x_batch, y_batch, label_batch_ = gen.rest_batch(config.batch_size)
                output_loss_, total_loss_, yhat_ = dev_step(x_batch, y_batch)
                for n in range(config.epoch_seq_len):
                    yhat[n, (test_step-1)*10*config.batch_size : len(gen.data_index)] = yhat_[n]
                output_loss += output_loss_
                total_loss += total_loss_
            yhat = yhat + 1
            acc = 0
            with open(os.path.join(out_dir, log_filename), "a") as text_file:
                text_file.write("{:g} {:g} ".format(output_loss, total_loss))
                for n in range(config.epoch_seq_len):
                    #print('n: {:g}\n'.format(n))
                    #print('yhat')
                    #print(yhat[n,:])
                    #print('y')
                    #print(gen.label[gen.data_index - (config.epoch_seq_len - 1) + n])
                    acc_n = accuracy_score(yhat[n,:], gen.label[gen.data_index - (config.epoch_seq_len - 1) + n]) # due to zero-indexing
                    if n == config.epoch_seq_len - 1:
                        text_file.write("{:g} \n".format(acc_n))
                    else:
                        text_file.write("{:g} ".format(acc_n))
                    acc += acc_n
            acc /= config.epoch_seq_len
            return acc, yhat, output_loss, total_loss

        # test off the pretrained model (no finetuning whatsoever at this point)
        print("{} Start off validation".format(datetime.now()))
        eval_acc, eval_yhat, eval_output_loss, eval_total_loss = evaluate(gen=eval_generator, log_filename="eval_result_log.txt")
        #test_acc, test_yhat, test_output_loss, test_total_loss = evaluate(gen=test_generator, log_filename="test_result_log.txt")
        #test_generator.reset_pointer()
        eval_generator.reset_pointer()

        start_time = time.time()
        # Loop over number of epochs
        for epoch in range(config.training_epoch):
            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
            step = 1
            while step < train_batches_per_epoch:
                # Get a batch
                x_batch, y_batch, label_batch = train_generator.next_batch(config.batch_size)
                train_step_, train_output_loss_, train_total_loss_, train_acc_ = train_step(x_batch, y_batch)
                time_str = datetime.now().isoformat()

                # average acc over sequences
                acc_ = 0
                for n in range(config.epoch_seq_len):
                    acc_ += train_acc_[n]
                acc_ /= config.epoch_seq_len

                print("{}: step {}, output_loss {}, total_loss {} acc {}".format(time_str, train_step_, train_output_loss_, train_total_loss_, acc_))
                with open(os.path.join(out_dir, "train_log.txt"), "a") as text_file:
                    text_file.write("{:g} {:g} {:g} {:g}\n".format(train_step_, train_output_loss_, train_total_loss_, acc_))
                step += 1

                current_step = tf.train.global_step(sess, global_step)
                if current_step % config.evaluate_every == 0:
                    # Validate the model on the validation and test sets
                    print("{} Start validation".format(datetime.now()))
                    eval_acc, eval_yhat, eval_output_loss, eval_total_loss = evaluate(gen=eval_generator, log_filename="eval_result_log.txt")
                    #test_acc, test_yhat, test_output_loss, test_total_loss = evaluate(gen=test_generator, log_filename="test_result_log.txt")

                    early_stop_count += 1
                    if(eval_acc >= best_acc):
                        early_stop_count = 0 # reset
                        best_acc = eval_acc
                        checkpoint_name = os.path.join(checkpoint_path, 'model_step' + str(current_step) +'.ckpt')
                        save_path = saver.save(sess, checkpoint_name)

                        print("Best model updated")
                        source_file = checkpoint_name
                        dest_file = os.path.join(checkpoint_path, 'best_model_acc')
                        shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                        shutil.copy(source_file + '.index', dest_file + '.index')
                        shutil.copy(source_file + '.meta', dest_file + '.meta')

                        # write current best performance to file
                        with open(os.path.join(out_dir, "current_best.txt"), "a") as text_file:
                            #text_file.write("Validation accuracy {:g} \n".format(eval_acc))
                            #text_file.write("Test accuracy {:g} \n".format(test_acc))
                            #text_file.write("{:g} {:g}\n".format(eval_acc, test_acc))
                            text_file.write("{:g}\n".format(eval_acc))

                    #test_generator.reset_pointer()
                    eval_generator.reset_pointer()

                    if(FLAGS.early_stopping == True):
                        print('EARLY STOPPING enabled!')
                        # early stop after 10 training steps without improvement.
                        if(early_stop_count >= config.early_stop_count):
                            end_time = time.time()
                            with open(os.path.join(out_dir, "training_time.txt"), "a") as text_file:
                                text_file.write("{:g}\n".format((end_time - start_time)))
                            quit()
                    else:
                        print('EARLY STOPPING disabled!')

            train_generator.reset_pointer()
            train_generator.shuffle_data()

        end_time = time.time()

        with open(os.path.join(out_dir, "training_time.txt"), "a") as text_file:
            text_file.write("{:g}\n".format((end_time - start_time)))
