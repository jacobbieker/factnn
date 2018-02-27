# Import of every needed library
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import pickle
import h5py
import gzip
import time
import csv
import sys
import os


path_mc_images = "/tree/tf/00_MC_Images.h5"
save_model_path = "/notebooks/thesis/jan/hyperModels/"

model_names = ['cf' 'cff' 'cfff' 'cffff' 'cfffff' 'ccf' 'ccff' 'ccfff' 'ccffff' 'ccfffff' 'cccf' 'cccff' 'cccfff' 'cccffff' 'cccfffff' 'ccccf' 'ccccff' 'ccccfff' 'ccccffff' 'ccccfffff' 'cccccf' 'cccccff' 'cccccfff' 'cccccffff' 'cccccfffff' 'ccccccf' 'ccccccff' 'ccccccfff' 'ccccccffff' 'ccccccfffff']

for model_name in model_names:
    # Number of files in validation-/test-dataset
    num_events = 200000
    events_in_validation = 10000
    # Number of nets to compute
    number_of_nets = 50
    # Comment on the run
    title_name = 'Compare_diffuse_flat_dropout_architectures'

    # Iterates over all existing models and chooses the right folder to save everything 
    file_paths = os.listdir(save_model_path)
    for path in file_paths:
        name = '_' + model_name
        if path.endswith(name):
            correct_path = path 

    # Creates missing folders or chooses the right one to append new data to
    if 'correct_path' in locals():
        folder_path = os.path.join(save_model_path, correct_path)
    else:
        folder_number = len(os.listdir(save_model_path))+1
        folder_path = save_model_path + '/' + str(folder_number) + '_' + model_name
        os.mkdir(folder_path)

        # Creates the csv to save every models performance in
        c_count = model_name.count('c')
        depth_names = []
        for i in range(c_count):
            depth_names.append('Depth_{}'.format(i+1))
        columns = ['Learning_Rate','Batch_Size','Patch_Size']
        columns.extend(depth_names)
        columns.extend(['Hidden_Nodes','Accuracy','Auc','Steps', 'Early_Stopped','Time', 'Title'])

        with open(os.path.join(folder_path, model_name+'_Hyperparameter.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(columns)



    def metaYielder():
        with h5py.File(path_mc_images, 'r') as f:
            keys = list(f.keys())
            events = []
            for key in keys:
                events.append(len(f[key]))

        gamma_anteil = events[0]/np.sum(events)
        hadron_anteil = events[1]/np.sum(events)

        gamma_count = int(round(num_events*gamma_anteil))
        hadron_count = int(round(num_events*hadron_anteil))

        return gamma_anteil, hadron_anteil, gamma_count, hadron_count


    def batchYielder():
        gamma_anteil, hadron_anteil, gamma_count, hadron_count = metaYielder()

        gamma_batch_size = int(round(batch_size*gamma_anteil))
        hadron_batch_size = int(round(batch_size*hadron_anteil))

        for step in range(num_steps):
            gamma_offset = (step * gamma_batch_size) % (gamma_count - gamma_batch_size)
            hadron_offset = (step * hadron_batch_size) % (hadron_count - hadron_batch_size)

            with h5py.File(path_mc_images, 'r') as f:
                gamma_data = f['Gamma'][gamma_offset:(gamma_offset + gamma_batch_size), :, :, :]
                hadron_data = f['Hadron'][hadron_offset:(hadron_offset + hadron_batch_size), :, :, :]

            batch_data = np.concatenate((gamma_data, hadron_data), axis=0)
            labels = np.array([True]*gamma_batch_size+[False]*hadron_batch_size)
            batch_labels = (np.arange(2) == labels[:,None]).astype(np.float32)

            yield batch_data, batch_labels



    gamma_anteil, hadron_anteil, gamma_count, hadron_count = metaYielder()

    with h5py.File(path_mc_images, 'r') as f:
        gamma_size = int(round(events_in_validation*gamma_anteil))
        hadron_size = int(round(events_in_validation*hadron_anteil))

        gamma_valid_data = f['Gamma'][gamma_count:(gamma_count+gamma_size), :, :, :]
        hadron_valid_data = f['Hadron'][hadron_count:(hadron_count+hadron_size), :, :, :]

        valid_dataset = np.concatenate((gamma_valid_data, hadron_valid_data), axis=0)
        labels = np.array([True]*gamma_size+[False]*hadron_size)
        valid_labels = (np.arange(2) == labels[:,None]).astype(np.float32)


        gamma_test_data = f['Gamma'][(gamma_count+gamma_size):(gamma_count+2*gamma_size), :, :, :]
        hadron_test_data = f['Hadron'][(hadron_count+hadron_size):(hadron_count+2*hadron_size), :, :, :]

        test_dataset = np.concatenate((gamma_test_data, hadron_test_data), axis=0)
        labels = np.array([True]*gamma_size+[False]*hadron_size)
        test_labels = (np.arange(2) == labels[:,None]).astype(np.float32)




    # Hyperparameter for the model (fit manually)
    num_labels = 2 # gamma or proton
    num_channels = 1 # it is a greyscale image

    # Maximum batches for the model
    num_steps = [20001] * number_of_nets
    # Generic learning-rate fot the adam optimzier
    learning_rate = [0.001] * number_of_nets
    # How many images will be in a batch
    batch_size = np.random.randint(64, 257, size=number_of_nets) # 64 - 257
    # Will the kernel/patch be 3x3 or 5x5
    patch_size = np.random.randint(0, 2, size=number_of_nets)*2+3 # 3 / 5
    # Setting the depth of the convolution layers. New layers will be longer than the preceding
    min_depth = 2
    max_depth = 21
    layer = model_name[:-1]

    depth = []
    if layer and layer[0]=='c':
        layer = layer[1:]
        depth.append(np.random.randint(min_depth, max_depth, size=number_of_nets)) # 2 - 21
    if layer and layer[0]=='c':
        layer = layer[1:]
        depth.append(np.random.randint(min_depth, max_depth, size=number_of_nets) + depth[0])
    if layer and layer[0]=='c':
        layer = layer[1:]
        depth.append(np.random.randint(min_depth, max_depth, size=number_of_nets) + depth[1])
    if layer and layer[0]=='c':
        layer = layer[1:]
        depth.append(np.random.randint(min_depth, max_depth, size=number_of_nets) + depth[2])
    if layer and layer[0]=='c':
        layer = layer[1:]
        depth.append(np.random.randint(min_depth, max_depth, size=number_of_nets) + depth[3])
    if layer and layer[0]=='c':
        layer = layer[1:]
        depth.append(np.random.randint(min_depth, max_depth, size=number_of_nets) + depth[4])

    # Number of hidden nodes in f-layers. all f-layers will have the same number of nodes
    num_hidden = np.random.randint(8, 257, size=number_of_nets) # 8 - 257

    # Combining the hyperparameters to fit them into a for-loop
    hyperparameter = zip(num_steps, learning_rate, batch_size, patch_size, zip(*depth), num_hidden)

    # Loading the existing runs to find the best auc till now. Only a model with a better auc will be saved
    df = pd.read_csv(os.path.join(folder_path, model_name+'_Hyperparameter.csv'))
    if len(df['Auc']) > 0:
        best_auc = df['Auc'].max()
    else:
        best_auc = 0






    # Main loop with the training process
    for num_steps, learning_rate, batch_size, patch_size, depth, num_hidden in hyperparameter:    
        try:
            # Measuring the loop-time
            start = time.time()
            # Path to logfiles and correct file name
            LOGDIR = '/tree/tf/cnn_logs'
            # Getting the right count-number for the new logfiles
            logcount = str(len(os.listdir(LOGDIR)))
            hparams = '_bs={}_ps={}_d={}_nh={}_ns={}'.format(batch_size, patch_size, depth, num_hidden, num_steps)


            # Setting the restrictions for tensorflow to not consume every resource it can find
            gpu_config = tf.GPUOptions(allocator_type='BFC')
            session_conf = tf.ConfigProto(gpu_options=gpu_config, intra_op_parallelism_threads=12, inter_op_parallelism_threads=12)
            tf.reset_default_graph()
            sess = tf.Session(config=session_conf)


            # Create tf.variables for the three different datasets
            tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 46, 45, num_channels), name='train_data')
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='train_labels')

            tf_valid_dataset = tf.constant(valid_dataset, name='valid_data')
            tf_valid_labels = tf.constant(valid_labels, name='valid_labels')

            tf_test_dataset_final = tf.constant(test_dataset, name='test_data_final')
            tf_test_labels_final = tf.constant(test_labels, name='test_labels_final')                    

            # Summary for same example input images
            tf.summary.image('input', tf_train_dataset, 6)



            # Creating the graph. Only layers specified in 'model_name' will be added and correctly sized
            layer = model_name[:-1]

            if layer and layer[0]=='c':
                layer = layer[1:]
                with tf.name_scope('conv2d_1'):
                    conv2d_1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth[0]], stddev=0.1), name='W')
                    conv2d_1_biases = tf.Variable(tf.constant(1.0, shape=[depth[0]]), name='B')

                    conv = tf.nn.conv2d(tf_train_dataset, conv2d_1_weights, [1, 1, 1, 1], padding='SAME')
                    hidden = tf.nn.relu(conv + conv2d_1_biases)
                    pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                    pool = tf.nn.dropout(pool, 0.9)

                    tf.summary.histogram("weights", conv2d_1_weights)
                    tf.summary.histogram("biases", conv2d_1_biases)
                    tf.summary.histogram("activations", hidden)
                    tf.summary.histogram("pooling", pool)



            if layer and layer[0]=='c':
                layer = layer[1:]        
                with tf.name_scope('conv2d_2'):
                    conv2d_2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth[0], depth[1]], stddev=0.1), name='W')
                    conv2d_2_biases = tf.Variable(tf.constant(1.0, shape=[depth[1]]), name='B')

                    conv = tf.nn.conv2d(pool, conv2d_2_weights, [1, 1, 1, 1], padding='SAME') 
                    hidden = tf.nn.relu(conv + conv2d_2_biases)
                    pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                    pool = tf.nn.dropout(pool, 0.9)

                    tf.summary.histogram("weights", conv2d_2_weights)
                    tf.summary.histogram("biases", conv2d_2_biases)
                    tf.summary.histogram("activations", hidden)
                    tf.summary.histogram("pooling", pool)



            if layer and layer[0]=='c':
                layer = layer[1:]        
                with tf.name_scope('conv2d_3'):
                    conv2d_3_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth[1], depth[2]], stddev=0.1), name='W')
                    conv2d_3_biases = tf.Variable(tf.constant(1.0, shape=[depth[2]]), name='B')

                    conv = tf.nn.conv2d(pool, conv2d_3_weights, [1, 1, 1, 1], padding='SAME') 
                    hidden = tf.nn.relu(conv + conv2d_3_biases)
                    pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                    pool = tf.nn.dropout(pool, 0.9)

                    tf.summary.histogram("weights", conv2d_3_weights)
                    tf.summary.histogram("biases", conv2d_3_biases)
                    tf.summary.histogram("activations", hidden)
                    tf.summary.histogram("pooling", pool)



            if layer and layer[0]=='c':
                layer = layer[1:]        
                with tf.name_scope('conv2d_4'):
                    conv2d_4_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth[2], depth[3]], stddev=0.1), name='W')
                    conv2d_4_biases = tf.Variable(tf.constant(1.0, shape=[depth[3]]), name='B')

                    conv = tf.nn.conv2d(pool, conv2d_4_weights, [1, 1, 1, 1], padding='SAME') 
                    hidden = tf.nn.relu(conv + conv2d_4_biases)
                    pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                    pool = tf.nn.dropout(pool, 0.9)

                    tf.summary.histogram("weights", conv2d_4_weights)
                    tf.summary.histogram("biases", conv2d_4_biases)
                    tf.summary.histogram("activations", hidden)
                    tf.summary.histogram("pooling", pool)



            if layer and layer[0]=='c':
                layer = layer[1:]        
                with tf.name_scope('conv2d_5'):
                    conv2d_5_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth[3], depth[4]], stddev=0.1), name='W')
                    conv2d_5_biases = tf.Variable(tf.constant(1.0, shape=[depth[4]]), name='B')

                    conv = tf.nn.conv2d(pool, conv2d_5_weights, [1, 1, 1, 1], padding='SAME') 
                    hidden = tf.nn.relu(conv + conv2d_5_biases)
                    pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                    pool = tf.nn.dropout(pool, 0.9)

                    tf.summary.histogram("weights", conv2d_5_weights)
                    tf.summary.histogram("biases", conv2d_5_biases)
                    tf.summary.histogram("activations", hidden)
                    tf.summary.histogram("pooling", pool)



            if layer and layer[0]=='c':
                layer = layer[1:]        
                with tf.name_scope('conv2d_6'):
                    conv2d_6_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth[4], depth[5]], stddev=0.1), name='W')
                    conv2d_6_biases = tf.Variable(tf.constant(1.0, shape=[depth[5]]), name='B')

                    conv = tf.nn.conv2d(pool, conv2d_6_weights, [1, 1, 1, 1], padding='SAME') 
                    hidden = tf.nn.relu(conv + conv2d_6_biases)
                    pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                    pool = tf.nn.dropout(pool, 0.9)

                    tf.summary.histogram("weights", conv2d_6_weights)
                    tf.summary.histogram("biases", conv2d_6_biases)
                    tf.summary.histogram("activations", hidden)                
                    tf.summary.histogram("pooling", pool)



            # Reshape convolution layers to process the nodes further with connected layers
            with tf.name_scope('reshape'):
                shape = pool.get_shape().as_list()
                output = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
                output = tf.nn.dropout(output, 0.75)



            if layer and layer[0]=='f':
                layer = layer[1:]        
                with tf.name_scope('fc_1'):
                    shape = output.get_shape().as_list()
                    fc_1_weights = tf.Variable(tf.truncated_normal([shape[1], num_hidden], stddev=0.1), name='W')
                    fc_1_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]), name='B')

                    output = tf.nn.relu(tf.matmul(output, fc_1_weights) + fc_1_biases)
                    output = tf.nn.dropout(output, 0.5)

                    tf.summary.histogram("weights", fc_1_weights)
                    tf.summary.histogram("biases", fc_1_biases)
                    tf.summary.histogram("activations", output)



            if layer and layer[0]=='f':
                layer = layer[1:]        
                with tf.name_scope('fc_2'):
                    shape = output.get_shape().as_list()
                    fc_2_weights = tf.Variable(tf.truncated_normal([shape[1], num_hidden], stddev=0.1), name='W')
                    fc_2_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]), name='B')

                    output = tf.nn.relu(tf.matmul(output, fc_2_weights) + fc_2_biases)
                    output = tf.nn.dropout(output, 0.5)

                    tf.summary.histogram("weights", fc_2_weights)
                    tf.summary.histogram("biases", fc_2_biases)
                    tf.summary.histogram("activations", output)



            if layer and layer[0]=='f':
                layer = layer[1:]        
                with tf.name_scope('fc_3'):
                    shape = output.get_shape().as_list()
                    fc_3_weights = tf.Variable(tf.truncated_normal([shape[1], num_hidden], stddev=0.1), name='W')
                    fc_3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]), name='B')

                    output = tf.nn.relu(tf.matmul(output, fc_3_weights) + fc_3_biases)
                    output = tf.nn.dropout(output, 0.5)

                    tf.summary.histogram("weights", fc_3_weights)
                    tf.summary.histogram("biases", fc_3_biases)
                    tf.summary.histogram("activations", output)



            if layer and layer[0]=='f':
                layer = layer[1:]        
                with tf.name_scope('fc_4'):
                    shape = output.get_shape().as_list()
                    fc_4_weights = tf.Variable(tf.truncated_normal([shape[1], num_hidden], stddev=0.1), name='W')
                    fc_4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]), name='B')

                    output = tf.nn.relu(tf.matmul(output, fc_4_weights) + fc_4_biases)
                    output = tf.nn.dropout(output, 0.5)

                    tf.summary.histogram("weights", fc_4_weights)
                    tf.summary.histogram("biases", fc_4_biases)
                    tf.summary.histogram("activations", output)




            with tf.name_scope('output'):
                shape = output.get_shape().as_list()
                output_weights = tf.Variable(tf.truncated_normal([shape[1], num_labels], stddev=0.1), name='W')
                output_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='B')

                output = tf.matmul(output, output_weights) + output_biases

                tf.summary.histogram("weights", output_weights)
                tf.summary.histogram("biases", output_biases)
                tf.summary.histogram("activations", output)



            # Computing the loss of the model
            with tf.name_scope('loss'):
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=tf_train_labels), name='loss')
                tf.summary.scalar("loss", loss)

            # Optimizing the model
            with tf.name_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

            # Predictions for the training, validation, and test data
            with tf.name_scope('prediction'):
                train_prediction = tf.nn.softmax(output)

            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(train_prediction, 1), tf.argmax(tf_train_labels, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('batch_accuracy', accuracy)



            # Computing the validation-dataset
            with tf.name_scope('validation'):
                layer = model_name[:-1]

                if layer and layer[0]=='c':
                    layer = layer[1:]
                    pool = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(tf_valid_dataset, conv2d_1_weights, [1, 1, 1, 1], padding='SAME') + conv2d_1_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                if layer and layer[0]=='c':
                    layer = layer[1:]
                    pool = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool, conv2d_2_weights, [1, 1, 1, 1], padding='SAME')  + conv2d_2_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                if layer and layer[0]=='c':
                    layer = layer[1:]
                    pool = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool, conv2d_3_weights, [1, 1, 1, 1], padding='SAME')  + conv2d_3_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                if layer and layer[0]=='c':
                    layer = layer[1:]
                    pool = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool, conv2d_4_weights, [1, 1, 1, 1], padding='SAME')  + conv2d_4_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                if layer and layer[0]=='c':
                    layer = layer[1:]
                    pool = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool, conv2d_5_weights, [1, 1, 1, 1], padding='SAME')  + conv2d_5_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                if layer and layer[0]=='c':
                    layer = layer[1:]
                    pool = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool, conv2d_6_weights, [1, 1, 1, 1], padding='SAME')  + conv2d_6_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                shape = pool.get_shape().as_list()
                output = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
                if layer and layer[0]=='f':
                    layer = layer[1:]
                    output = tf.nn.relu(tf.matmul(output, fc_1_weights) + fc_1_biases)
                if layer and layer[0]=='f':
                    layer = layer[1:]
                    output = tf.nn.relu(tf.matmul(output, fc_2_weights) + fc_2_biases)
                if layer and layer[0]=='f':
                    layer = layer[1:]
                    output = tf.nn.relu(tf.matmul(output, fc_3_weights) + fc_3_biases)
                if layer and layer[0]=='f':
                    layer = layer[1:]
                    output = tf.nn.relu(tf.matmul(output, fc_4_weights) + fc_4_biases)
                valid_prediction = tf.nn.softmax(tf.matmul(output, output_weights) + output_biases)

                correct_prediction = tf.equal(tf.argmax(valid_prediction, 1), tf.argmax(valid_labels, 1))
                valid_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('validation_accuracy', valid_accuracy)                        

            with tf.name_scope('auc'):
                valid_auc = tf.metrics.auc(labels=tf_valid_labels, predictions=valid_prediction, curve='ROC')
                tf.summary.scalar('validation_auc_0', valid_auc[0])
                #tf.summary.scalar('validation_auc_1', valid_auc[1])



            # Computing the test-dataset
            with tf.name_scope('testing'):
                layer = model_name[:-1]

                if layer and layer[0]=='c':
                    layer = layer[1:]
                    pool = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(tf_test_dataset_final, conv2d_1_weights, [1, 1, 1, 1], padding='SAME') + conv2d_1_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                if layer and layer[0]=='c':
                    layer = layer[1:]
                    pool = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool, conv2d_2_weights, [1, 1, 1, 1], padding='SAME')  + conv2d_2_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                if layer and layer[0]=='c':
                    layer = layer[1:]
                    pool = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool, conv2d_3_weights, [1, 1, 1, 1], padding='SAME')  + conv2d_3_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                if layer and layer[0]=='c':
                    layer = layer[1:]
                    pool = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool, conv2d_4_weights, [1, 1, 1, 1], padding='SAME')  + conv2d_4_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                if layer and layer[0]=='c':
                    layer = layer[1:]
                    pool = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool, conv2d_5_weights, [1, 1, 1, 1], padding='SAME')  + conv2d_5_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                if layer and layer[0]=='c':
                    layer = layer[1:]
                    pool = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool, conv2d_6_weights, [1, 1, 1, 1], padding='SAME')  + conv2d_6_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                shape = pool.get_shape().as_list()
                output = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
                if layer and layer[0]=='f':
                    layer = layer[1:]
                    output = tf.nn.relu(tf.matmul(output, fc_1_weights) + fc_1_biases)
                if layer and layer[0]=='f':
                    layer = layer[1:]
                    output = tf.nn.relu(tf.matmul(output, fc_2_weights) + fc_2_biases)
                if layer and layer[0]=='f':
                    layer = layer[1:]
                    output = tf.nn.relu(tf.matmul(output, fc_3_weights) + fc_3_biases)
                if layer and layer[0]=='f':
                    layer = layer[1:]
                    output = tf.nn.relu(tf.matmul(output, fc_4_weights) + fc_4_biases)
                test_prediction = tf.nn.softmax(tf.matmul(output, output_weights) + output_biases)

                test_correct_prediction = tf.equal(tf.argmax(test_prediction, 1), tf.argmax(test_labels, 1))
                test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))
                tf.summary.scalar('validation_accuracy', test_accuracy)                        

            with tf.name_scope('auc'):
                test_auc = tf.metrics.auc(labels=tf_test_labels_final, predictions=test_prediction, curve='ROC')
                tf.summary.scalar('test_auc_0', test_auc[0])
                #tf.summary.scalar('validation_auc_1', test_auc[1])


            # Merge all summaries and create a saver
            summ = tf.summary.merge_all()
            saver = tf.train.Saver()

            # Initializing the model-variables and specify the logfiles
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            writer = tf.summary.FileWriter(LOGDIR+'/'+logcount+hparams)
            writer.add_graph(sess.graph)



            # Iterating over num_steps batches and train the model 
            gen = batchYielder()
            for step in range(num_steps):
                batch_data, batch_labels = next(gen)
                # Creating a feed_dict to train the model on in this step
                feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
                # Train the model for this step
                _, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict=feed_dict)



                # Updating the output to stay in touch with the training process
                # Checking for early-stopping with scikit-learn
                if (step % 100 == 0):
                    s = sess.run(summ, feed_dict={tf_train_dataset: batch_data, tf_train_labels: batch_labels})
                    writer.add_summary(s, step)

                    # Compute the accuracy and the roc-auc-score with scikit-learn
                    pred = sess.run(valid_prediction)
                    pred = np.array(list(zip(pred[:,0], pred[:,1])))
                    stop_acc = accuracy_score(np.argmax(valid_labels, axis=1), np.argmax(pred, axis=1))
                    stop_auc = roc_auc_score(valid_labels, pred)


                    # Check if early-stopping is necessary
                    auc_now = stop_auc
                    if step == 0:
                        stopping_auc = 0.0
                        sink_count = 0
                    else:
                        if auc_now > stopping_auc:
                            stopping_auc = auc_now
                            sink_count = 0
                            # Check if the model is better than the existing one and has to be saved
                            if stopping_auc > best_auc:
                                saver.save(sess, os.path.join(folder_path, model_name))
                                best_auc = stopping_auc
                        else:
                            sink_count += 1

                    # Printing a current evaluation of the model
                    print('St_auc: {}, sc: {},val: {}, Step: {}'.format(stopping_auc, sink_count, stop_acc*100, step))
                    if sink_count == 10:
                        break   


            # Compute the final score of the model       
            pred = sess.run(test_prediction)
            pred = np.array(list(zip(pred[:,0], pred[:,1])))
            f_acc = accuracy_score(np.argmax(test_labels, axis=1), np.argmax(pred, axis=1))
            f_auc = roc_auc_score(test_labels, pred)

            # Close the session
            sess.close()

            print('Final_auc: {}, Final_acc: {}'.format(f_auc, f_acc))
            # Save the run to the csv and restart the loop
            dauer = time.time() - start
            early_stopped = True if step < num_steps-1 else False
            with open(os.path.join(folder_path, model_name+'_Hyperparameter.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow([learning_rate, batch_size, patch_size, *depth, num_hidden, f_acc*100, f_auc, step, early_stopped, dauer, title_name])

        except:
            sess.close()
            raise
