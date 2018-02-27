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

def createFolders(model_name, save_model_path):
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
        columns.extend(['Accuracy','Auc','Pretraining_Steps', 'Title'])

        with open(os.path.join(folder_path, model_name+'_Hyperparameter.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            
    return folder_path



def metaYielder(path_mc_images):
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




def batchYielder(path_mc_images):
    gamma_anteil, hadron_anteil, gamma_count, hadron_count = metaYielder(path_mc_images)

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
        
        
        
        
def getValidationTesting(path_mc_images, events_in_validation_and_testing, gamma_anteil, hadron_anteil, gamma_count, hadron_count):
    with h5py.File(path_mc_images, 'r') as f:
        gamma_size = int(round(events_in_validation_and_testing*gamma_anteil))
        hadron_size = int(round(events_in_validation_and_testing*hadron_anteil))

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
        
    return valid_dataset, valid_labels, test_dataset, test_labels



def bestAuc(folder_path, architecture):
    # Loading the existing runs to find the best auc untill now. Only a model with a better auc will be saved
    df = pd.read_csv(os.path.join(folder_path, architecture+'_Hyperparameter.csv'))
    if len(df['Auc']) > 0:
        best_auc = df['Auc'].max()
    else:
        best_auc = 0
        
    return best_auc




def getHyperparameter(architecture, number_of_nets):
    # Hyperparameter for the model (fit manually)
    num_labels = 2 # gamma or proton
    num_channels = 1 # it is a greyscale image
    
    num_steps = 25001     # Maximum batches for the model
    
    min_batch_size = 64   # How many images will be in a batch
    max_batch_size = 257
    
    patch_size = [3, 5]   # Will the kernel/patch be 3x3 or 5x5

    min_depth = 2         # Setting the depth of the convolution layers. New layers will be longer than the preceding
    max_depth = 21
    
    min_num_hidden = 8    # Number of hidden nodes in f-layers. all f-layers will have the same number of nodes
    max_num_hidden = 257
    
    
    num_steps = [num_steps] * number_of_nets
    batch_size = np.random.randint(min_batch_size, max_batch_size, size=number_of_nets)
    patch_size = np.random.choice(patch_size, size=number_of_nets)
    layer = architecture[:-1]

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

    num_hidden = np.random.randint(min_num_hidden, max_num_hidden, size=number_of_nets)
    
    # Combining the hyperparameters to fit them into a for-loop
    hyperparameter = list(zip(num_steps, batch_size, patch_size, zip(*depth), num_hidden))
    
    return num_labels, num_channels, hyperparameter





def getSessConf(per_process_gpu_memory_fraction = 0.3, op_parallelism_threads = 18):
    gpu_config = tf.GPUOptions(allocator_type='BFC')
    session_conf = tf.ConfigProto(gpu_options=gpu_config, intra_op_parallelism_threads=op_parallelism_threads, inter_op_parallelism_threads=op_parallelism_threads)
    
    return session_conf




# Input arguments from outside
#path_mc_images = sys.argv[1]
#save_model_path = sys.argv[2]

path_mc_images = '/tree/tf/00_MC_Images.h5'
save_model_path = '/tree/tf/thesis/jan/hyperModels'

csv_path = '/tree/tf/pretraining.csv'
pickle_path = '/tree/tf/Pickle_{}.p'
if 'pretraining.csv' not in os.listdir('/tree/tf/'):
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Accuracy', 'Auc', 'Pretraining'])



def training(steps, best_auc):
    print('Layer {} training:'.format(pretraining_step))
    gen = batchYielder(path_mc_images)
    for step in range(steps):
        batch_data, batch_labels = next(gen)
        # Creating a feed_dict to train the model on in this step
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        # Train the model for this step
        _ = sess.run([optimizer], feed_dict=feed_dict)

        # Updating the output to stay in touch with the training process
        # Checking for early-stopping with scikit-learn
        if (step % 500 == 0):
            s = sess.run(summ, feed_dict={tf_train_dataset: batch_data, tf_train_labels: batch_labels})
            #writer.add_summary(s, step)

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
                        saver.save(sess, os.path.join(folder_path, architecture))
                        best_auc = stopping_auc
                else:
                    sink_count += 1
                    
            # Printing a current evaluation of the model
            print('St_auc: {}, sc: {},val: {}, Step: {}'.format(stopping_auc, sink_count, stop_acc*100, step))
            if sink_count == 10:
                break   

    return stop_acc, stopping_auc, step, best_auc






dropout_rate_c = 0.9
dropout_rate_c_output = 0.75
dropout_rate_f = 0.5

learning_rate = 0.001

pretraining_steps = [5001, 2001, 8001, 1001, 1]

# Number of events in training-dataset
num_events = 800000

# Number of events in validation-/test-dataset
events_in_validation_and_testing = 5000

# Number of nets to compute
number_of_nets = 30

trainable = True

# Architectures to test
test_architectures = ['ccccccffff']



gamma_anteil, hadron_anteil, gamma_count, hadron_count = metaYielder(path_mc_images)
valid_dataset, valid_labels, test_dataset, test_labels = getValidationTesting(path_mc_images, events_in_validation_and_testing, gamma_anteil, hadron_anteil, gamma_count, hadron_count)


for architecture in test_architectures:
    c_count = architecture.count('c')
    f_count = architecture.count('f')
    folder_path = createFolders(architecture, save_model_path)
    print('\n\n', folder_path)
    
    best_auc = bestAuc(folder_path, architecture)
        
    num_labels, num_channels, hyperparameter = getHyperparameter(architecture, number_of_nets)
    for pretraining in pretraining_steps:
        for num_steps, batch_size, patch_size, depth, num_hidden in hyperparameter:
            try:
                print(num_steps, batch_size, patch_size, depth, num_hidden)

                # Measuring the loop-time
                start = time.time()
                # Path to logfiles and correct file name
                LOGDIR = '/tree/tf/cnn_logs'
                # Getting the right count-number for the new logfiles
                logcount = str(len(os.listdir(LOGDIR)))
                hparams = '_bs={}_ps={}_d={}_nh={}_ns={}'.format(batch_size, patch_size, depth, num_hidden, num_steps)


                tf.reset_default_graph()
                with tf.Session(config=getSessConf()) as sess:
                    # Create tf.variables for the three different datasets
                    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 46, 45, num_channels), name='train_data')
                    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='train_labels')

                    tf_valid_dataset = tf.constant(valid_dataset, name='valid_data')
                    tf_valid_labels = tf.constant(valid_labels, name='valid_labels')

                    tf_test_dataset_final = tf.constant(test_dataset, name='test_data_final')
                    tf_test_labels_final = tf.constant(test_labels, name='test_labels_final')                    

                    # Summary for same example input images
                    tf.summary.image('input', tf_train_dataset, 6)


                        
                    weights_1 = []
                    biases_1 = []
                    pretraining_step = 1
                        
                    with tf.name_scope('{}_conv2d_1'.format(pretraining_step)):
                        layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth[0]], stddev=0.1), name='W_1')
                        layer1_biases = tf.Variable(tf.constant(1.0, shape=[depth[0]]), name='B_1')
                        conv = tf.nn.conv2d(tf_train_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer1_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)

                    # The reshape produces an input vector for the dense layer
                    with tf.name_scope('{}_reshape'.format(pretraining_step)):
                        shape = pool.get_shape().as_list()
                        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])

                    # Output layer is a dense layer
                    with tf.name_scope('{}_Output'.format(pretraining_step)):
                        output_weights = tf.Variable(tf.truncated_normal([23*23*depth[0], num_labels], stddev=0.1), name='W')
                        output_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='B')
                        output = tf.matmul(reshape, output_weights) + output_biases

                    # Computing the loss of the model
                    with tf.name_scope('{}_loss'.format(pretraining_step)):
                        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=tf_train_labels), name='loss')

                    # Optimizing the model
                    with tf.name_scope('{}_optimizer'.format(pretraining_step)):
                        optimizer = tf.train.AdamOptimizer(learning_rate, name='{}_adam'.format(pretraining_step)).minimize(loss)

                    # Predictions for the training, validation, and test data
                    with tf.name_scope('{}_prediction'.format(pretraining_step)):
                        train_prediction = tf.nn.softmax(output)

                    # Evaluating the network: accuracy
                    with tf.name_scope('{}_valid'.format(pretraining_step)):
                        pool_1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(tf_valid_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        shape = pool_1.get_shape().as_list()
                        reshape = tf.reshape(pool_1, [shape[0], shape[1] * shape[2] * shape[3]])
                        valid_prediction = tf.nn.softmax(tf.matmul(reshape, output_weights) + output_biases)

                        correct_prediction = tf.equal(tf.argmax(valid_prediction, 1), tf.argmax(tf_valid_labels, 1))
                        valid_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))                 
                        
                    # Evaluating the network: auc
                    with tf.name_scope('{}_auc'.format(pretraining_step)):
                        valid_auc = tf.metrics.auc(labels=tf_valid_labels, predictions=valid_prediction, curve='ROC')
                    print('Layers created')


                    summ = tf.summary.merge_all()
                    saver = tf.train.Saver()

                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    #writer = tf.summary.FileWriter(os.path.join(save_model_path, 'First_Layer'+hparams))
                    #writer.add_graph(sess.graph)

                    stop_acc, stopping_auc, step, best_auc = training(pretraining, best_auc)

                    weights_1.append(layer1_weights.eval())
                    biases_1.append(layer1_biases.eval())
                    
                    
                    
                    
                    weights_2 = []
                    biases_2 = []
                    pretraining_step = 2
                        
                    with tf.name_scope('{}_conv2d_1'.format(pretraining_step)):
                        init_w_1 = tf.constant(weights_1[0])
                        layer1_weights = tf.get_variable('W_1', initializer=init_w_1, trainable=trainable)
                        init_b_1 = tf.constant(biases_1[0])
                        layer1_biases = tf.get_variable('B_1', initializer=init_b_1, trainable=trainable)
                        conv = tf.nn.conv2d(tf_train_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer1_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_2'.format(pretraining_step)):
                        layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth[0], depth[1]], stddev=0.1), name='W_1')
                        layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth[1]]), name='B_1')
                        conv = tf.nn.conv2d(pool, layer2_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer2_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)

                    # The reshape produces an input vector for the dense layer
                    with tf.name_scope('{}_reshape'.format(pretraining_step)):
                        shape = pool.get_shape().as_list()
                        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])

                    # Output layer is a dense layer
                    with tf.name_scope('{}_Output'.format(pretraining_step)):
                        output_weights = tf.Variable(tf.truncated_normal([12*12*depth[1], num_labels], stddev=0.1), name='W')
                        output_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='B')
                        output = tf.matmul(reshape, output_weights) + output_biases

                    # Computing the loss of the model
                    with tf.name_scope('{}_loss'.format(pretraining_step)):
                        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=tf_train_labels), name='loss')

                    # Optimizing the model
                    with tf.name_scope('{}_optimizer'.format(pretraining_step)):
                        optimizer = tf.train.AdamOptimizer(learning_rate, name='{}_adam'.format(pretraining_step)).minimize(loss)

                    # Predictions for the training, validation, and test data
                    with tf.name_scope('{}_prediction'.format(pretraining_step)):
                        train_prediction = tf.nn.softmax(output)

                    # Evaluating the network: accuracy
                    with tf.name_scope('{}_valid'.format(pretraining_step)):
                        pool_1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(tf_valid_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_2 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_1, layer2_weights, [1, 1, 1, 1], padding='SAME') + layer2_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        shape = pool_2.get_shape().as_list()
                        reshape = tf.reshape(pool_2, [shape[0], shape[1] * shape[2] * shape[3]])
                        valid_prediction = tf.nn.softmax(tf.matmul(reshape, output_weights) + output_biases)

                        correct_prediction = tf.equal(tf.argmax(valid_prediction, 1), tf.argmax(tf_valid_labels, 1))
                        valid_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))                 
                        
                    # Evaluating the network: auc
                    with tf.name_scope('{}_auc'.format(pretraining_step)):
                        valid_auc = tf.metrics.auc(labels=tf_valid_labels, predictions=valid_prediction, curve='ROC')
                    print('Layers created')


                    summ = tf.summary.merge_all()
                    saver = tf.train.Saver()

                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    #writer = tf.summary.FileWriter(os.path.join(save_model_path, 'Second_Layer'+hparams))
                    #writer.add_graph(sess.graph)

                    stop_acc, stopping_auc, step, best_auc = training(pretraining, best_auc)

                    weights_2.append(layer1_weights.eval())
                    weights_2.append(layer2_weights.eval())
                    biases_2.append(layer1_biases.eval())
                    biases_2.append(layer2_biases.eval())
                    
                    
                    
                    
                    weights_3 = []
                    biases_3 = []
                    pretraining_step = 3
                        
                    with tf.name_scope('{}_conv2d_1'.format(pretraining_step)):
                        init_w_1 = tf.constant(weights_2[0])
                        layer1_weights = tf.get_variable('W_2', initializer=init_w_1, trainable=trainable)
                        init_b_1 = tf.constant(biases_2[0])
                        layer1_biases = tf.get_variable('B_2', initializer=init_b_1, trainable=trainable)
                        conv = tf.nn.conv2d(tf_train_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer1_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_2'.format(pretraining_step)):
                        init_w_2 = tf.constant(weights_2[1])
                        layer2_weights = tf.get_variable('W_3', initializer=init_w_2, trainable=trainable)
                        init_b_2 = tf.constant(biases_2[1])
                        layer2_biases = tf.get_variable('B_3', initializer=init_b_2, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer2_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer2_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_3'.format(pretraining_step)):
                        layer3_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth[1], depth[2]], stddev=0.1), name='W_1')
                        layer3_biases = tf.Variable(tf.constant(1.0, shape=[depth[2]]), name='B_1')
                        conv = tf.nn.conv2d(pool, layer3_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer3_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)

                    # The reshape produces an input vector for the dense layer
                    with tf.name_scope('{}_reshape'.format(pretraining_step)):
                        shape = pool.get_shape().as_list()
                        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])

                    # Output layer is a dense layer
                    with tf.name_scope('{}_Output'.format(pretraining_step)):
                        output_weights = tf.Variable(tf.truncated_normal([6*6*depth[2], num_labels], stddev=0.1), name='W')
                        output_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='B')
                        output = tf.matmul(reshape, output_weights) + output_biases

                    # Computing the loss of the model
                    with tf.name_scope('{}_loss'.format(pretraining_step)):
                        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=tf_train_labels), name='loss')

                    # Optimizing the model
                    with tf.name_scope('{}_optimizer'.format(pretraining_step)):
                        optimizer = tf.train.AdamOptimizer(learning_rate, name='{}_adam'.format(pretraining_step)).minimize(loss)

                    # Predictions for the training, validation, and test data
                    with tf.name_scope('{}_prediction'.format(pretraining_step)):
                        train_prediction = tf.nn.softmax(output)

                    # Evaluating the network: accuracy
                    with tf.name_scope('{}_valid'.format(pretraining_step)):
                        pool_1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(tf_valid_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_2 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_1, layer2_weights, [1, 1, 1, 1], padding='SAME') + layer2_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_3 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_2, layer3_weights, [1, 1, 1, 1], padding='SAME') + layer3_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        shape = pool_3.get_shape().as_list()
                        reshape = tf.reshape(pool_3, [shape[0], shape[1] * shape[2] * shape[3]])
                        valid_prediction = tf.nn.softmax(tf.matmul(reshape, output_weights) + output_biases)

                        correct_prediction = tf.equal(tf.argmax(valid_prediction, 1), tf.argmax(tf_valid_labels, 1))
                        valid_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))                 
                        
                    # Evaluating the network: auc
                    with tf.name_scope('{}_auc'.format(pretraining_step)):
                        valid_auc = tf.metrics.auc(labels=tf_valid_labels, predictions=valid_prediction, curve='ROC')
                    print('Layers created')


                    summ = tf.summary.merge_all()
                    saver = tf.train.Saver()

                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    #writer = tf.summary.FileWriter(os.path.join(save_model_path, 'Third_Layer'+hparams))
                    #writer.add_graph(sess.graph)

                    stop_acc, stopping_auc, step, best_auc = training(pretraining, best_auc)

                    weights_3.append(layer1_weights.eval())
                    weights_3.append(layer2_weights.eval())
                    weights_3.append(layer3_weights.eval())
                    biases_3.append(layer1_biases.eval())
                    biases_3.append(layer2_biases.eval())
                    biases_3.append(layer3_biases.eval())
                    
                    
                    
                    weights_4 = []
                    biases_4 = []
                    pretraining_step = 4
                        
                    with tf.name_scope('{}_conv2d_1'.format(pretraining_step)):
                        init_w_1 = tf.constant(weights_3[0])
                        layer1_weights = tf.get_variable('W_4', initializer=init_w_1, trainable=trainable)
                        init_b_1 = tf.constant(biases_3[0])
                        layer1_biases = tf.get_variable('B_4', initializer=init_b_1, trainable=trainable)
                        conv = tf.nn.conv2d(tf_train_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer1_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_2'.format(pretraining_step)):
                        init_w_2 = tf.constant(weights_3[1])
                        layer2_weights = tf.get_variable('W_5', initializer=init_w_2, trainable=trainable)
                        init_b_2 = tf.constant(biases_3[1])
                        layer2_biases = tf.get_variable('B_5', initializer=init_b_2, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer2_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer2_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_3'.format(pretraining_step)):
                        init_w_3 = tf.constant(weights_3[2])
                        layer3_weights = tf.get_variable('W_6', initializer=init_w_3, trainable=trainable)
                        init_b_3 = tf.constant(biases_3[2])
                        layer3_biases = tf.get_variable('B_6', initializer=init_b_3, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer3_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer3_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_4'.format(pretraining_step)):
                        layer4_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth[2], depth[3]], stddev=0.1), name='W_1')
                        layer4_biases = tf.Variable(tf.constant(1.0, shape=[depth[3]]), name='B_1')
                        conv = tf.nn.conv2d(pool, layer4_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer4_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)

                    # The reshape produces an input vector for the dense layer
                    with tf.name_scope('{}_reshape'.format(pretraining_step)):
                        shape = pool.get_shape().as_list()
                        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])

                    # Output layer is a dense layer
                    with tf.name_scope('{}_Output'.format(pretraining_step)):
                        output_weights = tf.Variable(tf.truncated_normal([3*3*depth[3], num_labels], stddev=0.1), name='W')
                        output_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='B')
                        output = tf.matmul(reshape, output_weights) + output_biases

                    # Computing the loss of the model
                    with tf.name_scope('{}_loss'.format(pretraining_step)):
                        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=tf_train_labels), name='loss')

                    # Optimizing the model
                    with tf.name_scope('{}_optimizer'.format(pretraining_step)):
                        optimizer = tf.train.AdamOptimizer(learning_rate, name='{}_adam'.format(pretraining_step)).minimize(loss)

                    # Predictions for the training, validation, and test data
                    with tf.name_scope('{}_prediction'.format(pretraining_step)):
                        train_prediction = tf.nn.softmax(output)

                    # Evaluating the network: accuracy
                    with tf.name_scope('{}_valid'.format(pretraining_step)):
                        pool_1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(tf_valid_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_2 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_1, layer2_weights, [1, 1, 1, 1], padding='SAME') + layer2_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_3 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_2, layer3_weights, [1, 1, 1, 1], padding='SAME') + layer3_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_4 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_3, layer4_weights, [1, 1, 1, 1], padding='SAME') + layer4_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        shape = pool_4.get_shape().as_list()
                        reshape = tf.reshape(pool_4, [shape[0], shape[1] * shape[2] * shape[3]])
                        valid_prediction = tf.nn.softmax(tf.matmul(reshape, output_weights) + output_biases)

                        correct_prediction = tf.equal(tf.argmax(valid_prediction, 1), tf.argmax(tf_valid_labels, 1))
                        valid_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))                 
                        
                    # Evaluating the network: auc
                    with tf.name_scope('{}_auc'.format(pretraining_step)):
                        valid_auc = tf.metrics.auc(labels=tf_valid_labels, predictions=valid_prediction, curve='ROC')
                    print('Layers created')


                    summ = tf.summary.merge_all()
                    saver = tf.train.Saver()

                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    #writer = tf.summary.FileWriter(os.path.join(save_model_path, 'Fourth_Layer'+hparams))
                    #writer.add_graph(sess.graph)

                    stop_acc, stopping_auc, step, best_auc = training(pretraining, best_auc)

                    weights_4.append(layer1_weights.eval())
                    weights_4.append(layer2_weights.eval())
                    weights_4.append(layer3_weights.eval())
                    weights_4.append(layer4_weights.eval())
                    biases_4.append(layer1_biases.eval())
                    biases_4.append(layer2_biases.eval())
                    biases_4.append(layer3_biases.eval())
                    biases_4.append(layer4_biases.eval())
                    
                    
                    
                    
                    weights_5 = []
                    biases_5 = []
                    pretraining_step = 5
                        
                    with tf.name_scope('{}_conv2d_1'.format(pretraining_step)):
                        init_w_1 = tf.constant(weights_4[0])
                        layer1_weights = tf.get_variable('W_7', initializer=init_w_1, trainable=trainable)
                        init_b_1 = tf.constant(biases_4[0])
                        layer1_biases = tf.get_variable('B_7', initializer=init_b_1, trainable=trainable)
                        conv = tf.nn.conv2d(tf_train_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer1_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_2'.format(pretraining_step)):
                        init_w_2 = tf.constant(weights_4[1])
                        layer2_weights = tf.get_variable('W_8', initializer=init_w_2, trainable=trainable)
                        init_b_2 = tf.constant(biases_4[1])
                        layer2_biases = tf.get_variable('B_8', initializer=init_b_2, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer2_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer2_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_3'.format(pretraining_step)):
                        init_w_3 = tf.constant(weights_4[2])
                        layer3_weights = tf.get_variable('W_9', initializer=init_w_3, trainable=trainable)
                        init_b_3 = tf.constant(biases_4[2])
                        layer3_biases = tf.get_variable('B_9', initializer=init_b_3, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer3_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer3_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_4'.format(pretraining_step)):
                        init_w_4 = tf.constant(weights_4[3])
                        layer4_weights = tf.get_variable('W_10', initializer=init_w_4, trainable=trainable)
                        init_b_4 = tf.constant(biases_4[3])
                        layer4_biases = tf.get_variable('B_10', initializer=init_b_4, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer4_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer4_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_5'.format(pretraining_step)):
                        layer5_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth[3], depth[4]], stddev=0.1), name='W_1')
                        layer5_biases = tf.Variable(tf.constant(1.0, shape=[depth[4]]), name='B_1')
                        conv = tf.nn.conv2d(pool, layer5_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer5_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)

                    # The reshape produces an input vector for the dense layer
                    with tf.name_scope('{}_reshape'.format(pretraining_step)):
                        shape = pool.get_shape().as_list()
                        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])

                    # Output layer is a dense layer
                    with tf.name_scope('{}_Output'.format(pretraining_step)):
                        output_weights = tf.Variable(tf.truncated_normal([2*2*depth[4], num_labels], stddev=0.1), name='W')
                        output_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='B')
                        output = tf.matmul(reshape, output_weights) + output_biases

                    # Computing the loss of the model
                    with tf.name_scope('{}_loss'.format(pretraining_step)):
                        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=tf_train_labels), name='loss')

                    # Optimizing the model
                    with tf.name_scope('{}_optimizer'.format(pretraining_step)):
                        optimizer = tf.train.AdamOptimizer(learning_rate, name='{}_adam'.format(pretraining_step)).minimize(loss)

                    # Predictions for the training, validation, and test data
                    with tf.name_scope('{}_prediction'.format(pretraining_step)):
                        train_prediction = tf.nn.softmax(output)

                    # Evaluating the network: accuracy
                    with tf.name_scope('{}_valid'.format(pretraining_step)):
                        pool_1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(tf_valid_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_2 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_1, layer2_weights, [1, 1, 1, 1], padding='SAME') + layer2_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_3 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_2, layer3_weights, [1, 1, 1, 1], padding='SAME') + layer3_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_4 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_3, layer4_weights, [1, 1, 1, 1], padding='SAME') + layer4_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_5 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_4, layer5_weights, [1, 1, 1, 1], padding='SAME') + layer5_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        shape = pool_5.get_shape().as_list()
                        reshape = tf.reshape(pool_5, [shape[0], shape[1] * shape[2] * shape[3]])
                        valid_prediction = tf.nn.softmax(tf.matmul(reshape, output_weights) + output_biases)

                        correct_prediction = tf.equal(tf.argmax(valid_prediction, 1), tf.argmax(tf_valid_labels, 1))
                        valid_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))                 
                        
                    # Evaluating the network: auc
                    with tf.name_scope('{}_auc'.format(pretraining_step)):
                        valid_auc = tf.metrics.auc(labels=tf_valid_labels, predictions=valid_prediction, curve='ROC')
                    print('Layers created')


                    summ = tf.summary.merge_all()
                    saver = tf.train.Saver()

                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    #writer = tf.summary.FileWriter(os.path.join(save_model_path, 'Fifth_Layer'+hparams))
                    #writer.add_graph(sess.graph)

                    stop_acc, stopping_auc, step, best_auc = training(pretraining, best_auc)

                    weights_5.append(layer1_weights.eval())
                    weights_5.append(layer2_weights.eval())
                    weights_5.append(layer3_weights.eval())
                    weights_5.append(layer4_weights.eval())
                    weights_5.append(layer5_weights.eval())
                    biases_5.append(layer1_biases.eval())
                    biases_5.append(layer2_biases.eval())
                    biases_5.append(layer3_biases.eval())
                    biases_5.append(layer4_biases.eval())
                    biases_5.append(layer5_biases.eval())
                    
                    
                    
                    
                    weights_6 = []
                    biases_6 = []
                    pretraining_step = 6
                        
                    with tf.name_scope('{}_conv2d_1'.format(pretraining_step)):
                        init_w_1 = tf.constant(weights_5[0])
                        layer1_weights = tf.get_variable('W_11', initializer=init_w_1, trainable=trainable)
                        init_b_1 = tf.constant(biases_5[0])
                        layer1_biases = tf.get_variable('B_11', initializer=init_b_1, trainable=trainable)
                        conv = tf.nn.conv2d(tf_train_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer1_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_2'.format(pretraining_step)):
                        init_w_2 = tf.constant(weights_5[1])
                        layer2_weights = tf.get_variable('W_12', initializer=init_w_2, trainable=trainable)
                        init_b_2 = tf.constant(biases_5[1])
                        layer2_biases = tf.get_variable('B_12', initializer=init_b_2, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer2_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer2_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_3'.format(pretraining_step)):
                        init_w_3 = tf.constant(weights_5[2])
                        layer3_weights = tf.get_variable('W_13', initializer=init_w_3, trainable=trainable)
                        init_b_3 = tf.constant(biases_5[2])
                        layer3_biases = tf.get_variable('B_13', initializer=init_b_3, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer3_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer3_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_4'.format(pretraining_step)):
                        init_w_4 = tf.constant(weights_5[3])
                        layer4_weights = tf.get_variable('W_14', initializer=init_w_4, trainable=trainable)
                        init_b_4 = tf.constant(biases_5[3])
                        layer4_biases = tf.get_variable('B_14', initializer=init_b_4, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer4_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer4_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_5'.format(pretraining_step)):
                        init_w_5 = tf.constant(weights_5[4])
                        layer5_weights = tf.get_variable('W_15', initializer=init_w_5, trainable=trainable)
                        init_b_5 = tf.constant(biases_5[4])
                        layer5_biases = tf.get_variable('B_15', initializer=init_b_5, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer5_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer5_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_6'.format(pretraining_step)):
                        layer6_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth[4], depth[5]], stddev=0.1), name='W_1')
                        layer6_biases = tf.Variable(tf.constant(1.0, shape=[depth[5]]), name='B_1')
                        conv = tf.nn.conv2d(pool, layer6_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer6_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c_output)

                    # The reshape produces an input vector for the dense layer
                    with tf.name_scope('{}_reshape'.format(pretraining_step)):
                        shape = pool.get_shape().as_list()
                        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])

                    # Output layer is a dense layer
                    with tf.name_scope('{}_Output'.format(pretraining_step)):
                        output_weights = tf.Variable(tf.truncated_normal([1*1*depth[5], num_labels], stddev=0.1), name='W')
                        output_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='B')
                        output = tf.matmul(reshape, output_weights) + output_biases

                    # Computing the loss of the model
                    with tf.name_scope('{}_loss'.format(pretraining_step)):
                        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=tf_train_labels), name='loss')

                    # Optimizing the model
                    with tf.name_scope('{}_optimizer'.format(pretraining_step)):
                        optimizer = tf.train.AdamOptimizer(learning_rate, name='{}_adam'.format(pretraining_step)).minimize(loss)

                    # Predictions for the training, validation, and test data
                    with tf.name_scope('{}_prediction'.format(pretraining_step)):
                        train_prediction = tf.nn.softmax(output)

                    # Evaluating the network: accuracy
                    with tf.name_scope('{}_valid'.format(pretraining_step)):
                        pool_1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(tf_valid_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_2 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_1, layer2_weights, [1, 1, 1, 1], padding='SAME') + layer2_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_3 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_2, layer3_weights, [1, 1, 1, 1], padding='SAME') + layer3_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_4 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_3, layer4_weights, [1, 1, 1, 1], padding='SAME') + layer4_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_5 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_4, layer5_weights, [1, 1, 1, 1], padding='SAME') + layer5_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_6 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_5, layer6_weights, [1, 1, 1, 1], padding='SAME') + layer6_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        shape = pool_6.get_shape().as_list()
                        reshape = tf.reshape(pool_6, [shape[0], shape[1] * shape[2] * shape[3]])
                        valid_prediction = tf.nn.softmax(tf.matmul(reshape, output_weights) + output_biases)

                        correct_prediction = tf.equal(tf.argmax(valid_prediction, 1), tf.argmax(tf_valid_labels, 1))
                        valid_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))                 
                        
                    # Evaluating the network: auc
                    with tf.name_scope('{}_auc'.format(pretraining_step)):
                        valid_auc = tf.metrics.auc(labels=tf_valid_labels, predictions=valid_prediction, curve='ROC')
                    print('Layers created')


                    summ = tf.summary.merge_all()
                    saver = tf.train.Saver()

                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    #writer = tf.summary.FileWriter(os.path.join(save_model_path, 'Sixth_Layer'+hparams))
                    #writer.add_graph(sess.graph)

                    stop_acc, stopping_auc, step, best_auc = training(pretraining, best_auc)

                    weights_6.append(layer1_weights.eval())
                    weights_6.append(layer2_weights.eval())
                    weights_6.append(layer3_weights.eval())
                    weights_6.append(layer4_weights.eval())
                    weights_6.append(layer5_weights.eval())
                    weights_6.append(layer6_weights.eval())
                    biases_6.append(layer1_biases.eval())
                    biases_6.append(layer2_biases.eval())
                    biases_6.append(layer3_biases.eval())
                    biases_6.append(layer4_biases.eval())
                    biases_6.append(layer5_biases.eval())
                    biases_6.append(layer6_biases.eval())
                    
                    
                    
                    
                    weights_7 = []
                    biases_7 = []
                    pretraining_step = 7
                        
                    with tf.name_scope('{}_conv2d_1'.format(pretraining_step)):
                        init_w_1 = tf.constant(weights_6[0])
                        layer1_weights = tf.get_variable('W_16', initializer=init_w_1, trainable=trainable)
                        init_b_1 = tf.constant(biases_6[0])
                        layer1_biases = tf.get_variable('B_16', initializer=init_b_1, trainable=trainable)
                        conv = tf.nn.conv2d(tf_train_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer1_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_2'.format(pretraining_step)):
                        init_w_2 = tf.constant(weights_6[1])
                        layer2_weights = tf.get_variable('W_17', initializer=init_w_2, trainable=trainable)
                        init_b_2 = tf.constant(biases_6[1])
                        layer2_biases = tf.get_variable('B_17', initializer=init_b_2, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer2_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer2_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_3'.format(pretraining_step)):
                        init_w_3 = tf.constant(weights_6[2])
                        layer3_weights = tf.get_variable('W_18', initializer=init_w_3, trainable=trainable)
                        init_b_3 = tf.constant(biases_6[2])
                        layer3_biases = tf.get_variable('B_18', initializer=init_b_3, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer3_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer3_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_4'.format(pretraining_step)):
                        init_w_4 = tf.constant(weights_6[3])
                        layer4_weights = tf.get_variable('W_19', initializer=init_w_4, trainable=trainable)
                        init_b_4 = tf.constant(biases_6[3])
                        layer4_biases = tf.get_variable('B_19', initializer=init_b_4, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer4_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer4_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_5'.format(pretraining_step)):
                        init_w_5 = tf.constant(weights_6[4])
                        layer5_weights = tf.get_variable('W_20', initializer=init_w_5, trainable=trainable)
                        init_b_5 = tf.constant(biases_6[4])
                        layer5_biases = tf.get_variable('B_20', initializer=init_b_5, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer5_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer5_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_6'.format(pretraining_step)):
                        init_w_6 = tf.constant(weights_6[5])
                        layer6_weights = tf.get_variable('W_21', initializer=init_w_6, trainable=trainable)
                        init_b_6 = tf.constant(biases_6[5])
                        layer6_biases = tf.get_variable('B_21', initializer=init_b_6, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer6_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer6_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c_output)

                    # The reshape produces an input vector for the dense layer
                    with tf.name_scope('{}_reshape'.format(pretraining_step)):
                        shape = pool.get_shape().as_list()
                        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
                        
                    with tf.name_scope('{}_fc_1'.format(pretraining_step)):
                        layer7_weights = tf.Variable(tf.truncated_normal([1*1*depth[5], num_hidden], stddev=0.1), name='W')
                        layer7_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]), name='B')
                        hidden = tf.nn.relu(tf.matmul(reshape, layer7_weights) + layer7_biases)
                        hidden = tf.nn.dropout(hidden, dropout_rate_f)

                    # Output layer is a dense layer
                    with tf.name_scope('{}_Output'.format(pretraining_step)):
                        output_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1), name='W')
                        output_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='B')
                        output = tf.matmul(hidden, output_weights) + output_biases

                    # Computing the loss of the model
                    with tf.name_scope('{}_loss'.format(pretraining_step)):
                        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=tf_train_labels), name='loss')

                    # Optimizing the model
                    with tf.name_scope('{}_optimizer'.format(pretraining_step)):
                        optimizer = tf.train.AdamOptimizer(learning_rate, name='{}_adam'.format(pretraining_step)).minimize(loss)

                    # Predictions for the training, validation, and test data
                    with tf.name_scope('{}_prediction'.format(pretraining_step)):
                        train_prediction = tf.nn.softmax(output)

                    # Evaluating the network: accuracy
                    with tf.name_scope('{}_valid'.format(pretraining_step)):
                        pool_1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(tf_valid_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_2 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_1, layer2_weights, [1, 1, 1, 1], padding='SAME') + layer2_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_3 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_2, layer3_weights, [1, 1, 1, 1], padding='SAME') + layer3_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_4 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_3, layer4_weights, [1, 1, 1, 1], padding='SAME') + layer4_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_5 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_4, layer5_weights, [1, 1, 1, 1], padding='SAME') + layer5_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_6 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_5, layer6_weights, [1, 1, 1, 1], padding='SAME') + layer6_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        shape = pool_6.get_shape().as_list()
                        reshape = tf.reshape(pool_6, [shape[0], shape[1] * shape[2] * shape[3]])
                        hidden_1 = tf.nn.relu(tf.matmul(reshape, layer7_weights) + layer7_biases)
                        valid_prediction = tf.nn.softmax(tf.matmul(hidden_1, output_weights) + output_biases)

                        correct_prediction = tf.equal(tf.argmax(valid_prediction, 1), tf.argmax(tf_valid_labels, 1))
                        valid_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))                 
                        
                    # Evaluating the network: auc
                    with tf.name_scope('{}_auc'.format(pretraining_step)):
                        valid_auc = tf.metrics.auc(labels=tf_valid_labels, predictions=valid_prediction, curve='ROC')
                    print('Layers created')


                    summ = tf.summary.merge_all()
                    saver = tf.train.Saver()

                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    #writer = tf.summary.FileWriter(os.path.join(save_model_path, 'Seventh_Layer'+hparams))
                    #writer.add_graph(sess.graph)

                    stop_acc, stopping_auc, step, best_auc = training(pretraining, best_auc)

                    weights_7.append(layer1_weights.eval())
                    weights_7.append(layer2_weights.eval())
                    weights_7.append(layer3_weights.eval())
                    weights_7.append(layer4_weights.eval())
                    weights_7.append(layer5_weights.eval())
                    weights_7.append(layer6_weights.eval())
                    weights_7.append(layer7_weights.eval())
                    biases_7.append(layer1_biases.eval())
                    biases_7.append(layer2_biases.eval())
                    biases_7.append(layer3_biases.eval())
                    biases_7.append(layer4_biases.eval())
                    biases_7.append(layer5_biases.eval())
                    biases_7.append(layer6_biases.eval())
                    biases_7.append(layer7_biases.eval())
                    
                    
                    
                    weights_8 = []
                    biases_8 = []
                    pretraining_step = 8
                        
                    with tf.name_scope('{}_conv2d_1'.format(pretraining_step)):
                        init_w_1 = tf.constant(weights_7[0])
                        layer1_weights = tf.get_variable('W_22', initializer=init_w_1, trainable=trainable)
                        init_b_1 = tf.constant(biases_7[0])
                        layer1_biases = tf.get_variable('B_22', initializer=init_b_1, trainable=trainable)
                        conv = tf.nn.conv2d(tf_train_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer1_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_2'.format(pretraining_step)):
                        init_w_2 = tf.constant(weights_7[1])
                        layer2_weights = tf.get_variable('W_23', initializer=init_w_2, trainable=trainable)
                        init_b_2 = tf.constant(biases_7[1])
                        layer2_biases = tf.get_variable('B_23', initializer=init_b_2, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer2_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer2_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_3'.format(pretraining_step)):
                        init_w_3 = tf.constant(weights_7[2])
                        layer3_weights = tf.get_variable('W_24', initializer=init_w_3, trainable=trainable)
                        init_b_3 = tf.constant(biases_7[2])
                        layer3_biases = tf.get_variable('B_24', initializer=init_b_3, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer3_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer3_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_4'.format(pretraining_step)):
                        init_w_4 = tf.constant(weights_7[3])
                        layer4_weights = tf.get_variable('W_25', initializer=init_w_4, trainable=trainable)
                        init_b_4 = tf.constant(biases_7[3])
                        layer4_biases = tf.get_variable('B_25', initializer=init_b_4, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer4_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer4_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_5'.format(pretraining_step)):
                        init_w_5 = tf.constant(weights_7[4])
                        layer5_weights = tf.get_variable('W_26', initializer=init_w_5, trainable=trainable)
                        init_b_5 = tf.constant(biases_7[4])
                        layer5_biases = tf.get_variable('B_26', initializer=init_b_5, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer5_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer5_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_6'.format(pretraining_step)):
                        init_w_6 = tf.constant(weights_7[5])
                        layer6_weights = tf.get_variable('W_27', initializer=init_w_6, trainable=trainable)
                        init_b_6 = tf.constant(biases_7[5])
                        layer6_biases = tf.get_variable('B_27', initializer=init_b_6, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer6_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer6_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c_output)

                    # The reshape produces an input vector for the dense layer
                    with tf.name_scope('{}_reshape'.format(pretraining_step)):
                        shape = pool.get_shape().as_list()
                        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
                        
                    with tf.name_scope('{}_fc_1'.format(pretraining_step)):
                        init_w_7 = tf.constant(weights_7[6])
                        layer7_weights = tf.get_variable('W_28', initializer=init_w_7, trainable=trainable)
                        init_b_7 = tf.constant(biases_7[6])
                        layer7_biases = tf.get_variable('B_28', initializer=init_b_7, trainable=trainable)
                        hidden = tf.nn.relu(tf.matmul(reshape, layer7_weights) + layer7_biases)
                        hidden = tf.nn.dropout(hidden, dropout_rate_f)
                        
                    with tf.name_scope('{}_fc_2'.format(pretraining_step)):
                        layer8_weights = tf.Variable(tf.truncated_normal([num_hidden, num_hidden], stddev=0.1), name='W')
                        layer8_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]), name='B')
                        hidden = tf.nn.relu(tf.matmul(hidden, layer8_weights) + layer8_biases)
                        hidden = tf.nn.dropout(hidden, dropout_rate_f)

                    # Output layer is a dense layer
                    with tf.name_scope('{}_Output'.format(pretraining_step)):
                        output_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1), name='W')
                        output_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='B')
                        output = tf.matmul(hidden, output_weights) + output_biases

                    # Computing the loss of the model
                    with tf.name_scope('{}_loss'.format(pretraining_step)):
                        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=tf_train_labels), name='loss')

                    # Optimizing the model
                    with tf.name_scope('{}_optimizer'.format(pretraining_step)):
                        optimizer = tf.train.AdamOptimizer(learning_rate, name='{}_adam'.format(pretraining_step)).minimize(loss)

                    # Predictions for the training, validation, and test data
                    with tf.name_scope('{}_prediction'.format(pretraining_step)):
                        train_prediction = tf.nn.softmax(output)

                    # Evaluating the network: accuracy
                    with tf.name_scope('{}_valid'.format(pretraining_step)):
                        pool_1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(tf_valid_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_2 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_1, layer2_weights, [1, 1, 1, 1], padding='SAME') + layer2_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_3 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_2, layer3_weights, [1, 1, 1, 1], padding='SAME') + layer3_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_4 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_3, layer4_weights, [1, 1, 1, 1], padding='SAME') + layer4_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_5 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_4, layer5_weights, [1, 1, 1, 1], padding='SAME') + layer5_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_6 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_5, layer6_weights, [1, 1, 1, 1], padding='SAME') + layer6_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        shape = pool_6.get_shape().as_list()
                        reshape = tf.reshape(pool_6, [shape[0], shape[1] * shape[2] * shape[3]])
                        hidden_1 = tf.nn.relu(tf.matmul(reshape, layer7_weights) + layer7_biases)
                        hidden_2 = tf.nn.relu(tf.matmul(hidden_1, layer8_weights) + layer8_biases)
                        valid_prediction = tf.nn.softmax(tf.matmul(hidden_2, output_weights) + output_biases)

                        correct_prediction = tf.equal(tf.argmax(valid_prediction, 1), tf.argmax(tf_valid_labels, 1))
                        valid_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))                 
                        
                    # Evaluating the network: auc
                    with tf.name_scope('{}_auc'.format(pretraining_step)):
                        valid_auc = tf.metrics.auc(labels=tf_valid_labels, predictions=valid_prediction, curve='ROC')
                    print('Layers created')


                    summ = tf.summary.merge_all()
                    saver = tf.train.Saver()

                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    #writer = tf.summary.FileWriter(os.path.join(save_model_path, 'Seventh_Layer'+hparams))
                    #writer.add_graph(sess.graph)

                    stop_acc, stopping_auc, step, best_auc = training(pretraining, best_auc)

                    weights_8.append(layer1_weights.eval())
                    weights_8.append(layer2_weights.eval())
                    weights_8.append(layer3_weights.eval())
                    weights_8.append(layer4_weights.eval())
                    weights_8.append(layer5_weights.eval())
                    weights_8.append(layer6_weights.eval())
                    weights_8.append(layer7_weights.eval())
                    weights_8.append(layer8_weights.eval())
                    biases_8.append(layer1_biases.eval())
                    biases_8.append(layer2_biases.eval())
                    biases_8.append(layer3_biases.eval())
                    biases_8.append(layer4_biases.eval())
                    biases_8.append(layer5_biases.eval())
                    biases_8.append(layer6_biases.eval())
                    biases_8.append(layer7_biases.eval())
                    biases_8.append(layer8_biases.eval())
                    
                    
                    
                    
                    weights_9 = []
                    biases_9 = []
                    pretraining_step = 9
                        
                    with tf.name_scope('{}_conv2d_1'.format(pretraining_step)):
                        init_w_1 = tf.constant(weights_8[0])
                        layer1_weights = tf.get_variable('W_29', initializer=init_w_1, trainable=trainable)
                        init_b_1 = tf.constant(biases_8[0])
                        layer1_biases = tf.get_variable('B_29', initializer=init_b_1, trainable=trainable)
                        conv = tf.nn.conv2d(tf_train_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer1_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_2'.format(pretraining_step)):
                        init_w_2 = tf.constant(weights_8[1])
                        layer2_weights = tf.get_variable('W_30', initializer=init_w_2, trainable=trainable)
                        init_b_2 = tf.constant(biases_8[1])
                        layer2_biases = tf.get_variable('B_30', initializer=init_b_2, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer2_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer2_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_3'.format(pretraining_step)):
                        init_w_3 = tf.constant(weights_8[2])
                        layer3_weights = tf.get_variable('W_31', initializer=init_w_3, trainable=trainable)
                        init_b_3 = tf.constant(biases_8[2])
                        layer3_biases = tf.get_variable('B_31', initializer=init_b_3, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer3_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer3_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_4'.format(pretraining_step)):
                        init_w_4 = tf.constant(weights_8[3])
                        layer4_weights = tf.get_variable('W_32', initializer=init_w_4, trainable=trainable)
                        init_b_4 = tf.constant(biases_8[3])
                        layer4_biases = tf.get_variable('B_32', initializer=init_b_4, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer4_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer4_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_5'.format(pretraining_step)):
                        init_w_5 = tf.constant(weights_8[4])
                        layer5_weights = tf.get_variable('W_33', initializer=init_w_5, trainable=trainable)
                        init_b_5 = tf.constant(biases_8[4])
                        layer5_biases = tf.get_variable('B_33', initializer=init_b_5, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer5_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer5_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c)
                        
                    with tf.name_scope('{}_conv2d_6'.format(pretraining_step)):
                        init_w_6 = tf.constant(weights_8[5])
                        layer6_weights = tf.get_variable('W_34', initializer=init_w_6, trainable=trainable)
                        init_b_6 = tf.constant(biases_8[5])
                        layer6_biases = tf.get_variable('B_34', initializer=init_b_6, trainable=trainable)
                        conv = tf.nn.conv2d(pool, layer6_weights, [1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv + layer6_biases)
                        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool = tf.nn.dropout(pool, dropout_rate_c_output)

                    # The reshape produces an input vector for the dense layer
                    with tf.name_scope('{}_reshape'.format(pretraining_step)):
                        shape = pool.get_shape().as_list()
                        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
                        
                    with tf.name_scope('{}_fc_1'.format(pretraining_step)):
                        init_w_7 = tf.constant(weights_8[6])
                        layer7_weights = tf.get_variable('W_35', initializer=init_w_7, trainable=trainable)
                        init_b_7 = tf.constant(biases_8[6])
                        layer7_biases = tf.get_variable('B_35', initializer=init_b_7, trainable=trainable)
                        hidden = tf.nn.relu(tf.matmul(reshape, layer7_weights) + layer7_biases)
                        hidden = tf.nn.dropout(hidden, dropout_rate_f)
                        
                    with tf.name_scope('{}_fc_2'.format(pretraining_step)):
                        init_w_8 = tf.constant(weights_8[7])
                        layer8_weights = tf.get_variable('W_36', initializer=init_w_8, trainable=trainable)
                        init_b_8 = tf.constant(biases_8[7])
                        layer8_biases = tf.get_variable('B_36', initializer=init_b_8, trainable=trainable)
                        hidden = tf.nn.relu(tf.matmul(hidden, layer8_weights) + layer8_biases)
                        hidden = tf.nn.dropout(hidden, dropout_rate_f)
                        
                    with tf.name_scope('{}_fc_3'.format(pretraining_step)):
                        layer9_weights = tf.Variable(tf.truncated_normal([num_hidden, num_hidden], stddev=0.1), name='W')
                        layer9_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]), name='B')
                        hidden = tf.nn.relu(tf.matmul(hidden, layer9_weights) + layer9_biases)
                        hidden = tf.nn.dropout(hidden, dropout_rate_f)

                    # Output layer is a dense layer
                    with tf.name_scope('{}_Output'.format(pretraining_step)):
                        output_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1), name='W')
                        output_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='B')
                        output = tf.matmul(hidden, output_weights) + output_biases

                    # Computing the loss of the model
                    with tf.name_scope('{}_loss'.format(pretraining_step)):
                        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=tf_train_labels), name='loss')

                    # Optimizing the model
                    with tf.name_scope('{}_optimizer'.format(pretraining_step)):
                        optimizer = tf.train.AdamOptimizer(learning_rate, name='{}_adam'.format(pretraining_step)).minimize(loss)

                    # Predictions for the training, validation, and test data
                    with tf.name_scope('{}_prediction'.format(pretraining_step)):
                        train_prediction = tf.nn.softmax(output)

                    # Evaluating the network: accuracy
                    with tf.name_scope('{}_valid'.format(pretraining_step)):
                        pool_1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(tf_valid_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_2 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_1, layer2_weights, [1, 1, 1, 1], padding='SAME') + layer2_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_3 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_2, layer3_weights, [1, 1, 1, 1], padding='SAME') + layer3_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_4 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_3, layer4_weights, [1, 1, 1, 1], padding='SAME') + layer4_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_5 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_4, layer5_weights, [1, 1, 1, 1], padding='SAME') + layer5_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_6 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_5, layer6_weights, [1, 1, 1, 1], padding='SAME') + layer6_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        shape = pool_6.get_shape().as_list()
                        reshape = tf.reshape(pool_6, [shape[0], shape[1] * shape[2] * shape[3]])
                        hidden_1 = tf.nn.relu(tf.matmul(reshape, layer7_weights) + layer7_biases)
                        hidden_2 = tf.nn.relu(tf.matmul(hidden_1, layer8_weights) + layer8_biases)
                        hidden_3 = tf.nn.relu(tf.matmul(hidden_2, layer9_weights) + layer9_biases)
                        valid_prediction = tf.nn.softmax(tf.matmul(hidden_3, output_weights) + output_biases)

                        correct_prediction = tf.equal(tf.argmax(valid_prediction, 1), tf.argmax(tf_valid_labels, 1))
                        valid_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))                 
                        
                    # Evaluating the network: auc
                    with tf.name_scope('{}_auc'.format(pretraining_step)):
                        valid_auc = tf.metrics.auc(labels=tf_valid_labels, predictions=valid_prediction, curve='ROC')
                    
                    # Evaluating the network: accuracy
                    with tf.name_scope('{}_valid'.format(pretraining_step)):
                        pool_1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(tf_test_dataset_final, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_2 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_1, layer2_weights, [1, 1, 1, 1], padding='SAME') + layer2_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_3 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_2, layer3_weights, [1, 1, 1, 1], padding='SAME') + layer3_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_4 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_3, layer4_weights, [1, 1, 1, 1], padding='SAME') + layer4_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_5 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_4, layer5_weights, [1, 1, 1, 1], padding='SAME') + layer5_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        pool_6 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_5, layer6_weights, [1, 1, 1, 1], padding='SAME') + layer6_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        shape = pool_6.get_shape().as_list()
                        reshape = tf.reshape(pool_6, [shape[0], shape[1] * shape[2] * shape[3]])
                        hidden_1 = tf.nn.relu(tf.matmul(reshape, layer7_weights) + layer7_biases)
                        hidden_2 = tf.nn.relu(tf.matmul(hidden_1, layer8_weights) + layer8_biases)
                        hidden_3 = tf.nn.relu(tf.matmul(hidden_2, layer9_weights) + layer9_biases)
                        test_prediction = tf.nn.softmax(tf.matmul(hidden_3, output_weights) + output_biases)

                        test_correct_prediction = tf.equal(tf.argmax(test_prediction, 1), tf.argmax(tf_test_labels_final, 1))
                        test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))                 
                        
                    # Evaluating the network: auc
                    with tf.name_scope('{}_auc'.format(pretraining_step)):
                        test_auc = tf.metrics.auc(labels=tf_test_labels_final, predictions=test_prediction, curve='ROC')
                    print('Layers created')


                    summ = tf.summary.merge_all()
                    saver = tf.train.Saver()

                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    #writer = tf.summary.FileWriter(os.path.join(save_model_path, 'Nineth_Layer'+hparams))
                    #writer.add_graph(sess.graph)
                    
                    stop_acc, stopping_auc, step, best_auc = training(num_steps, best_auc)
                    
                    pred = sess.run(test_prediction)
                    pred = np.array(list(zip(pred[:,0], pred[:,1])))
                    stop_acc = accuracy_score(np.argmax(test_labels, axis=1), np.argmax(pred, axis=1))
                    stop_auc = roc_auc_score(test_labels, pred)
                    print('Final AUC: {}, Final Acc: {}'.format(stop_auc, stop_acc))
                    
                    with open(csv_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([stop_acc, stop_auc, pretraining])

                    weights_9.append(layer1_weights.eval())
                    weights_9.append(layer2_weights.eval())
                    weights_9.append(layer3_weights.eval())
                    weights_9.append(layer4_weights.eval())
                    weights_9.append(layer5_weights.eval())
                    weights_9.append(layer6_weights.eval())
                    weights_9.append(layer7_weights.eval())
                    weights_9.append(layer8_weights.eval())
                    weights_9.append(layer9_weights.eval())
                    weights_9.append(output_weights.eval())
                    biases_9.append(layer1_biases.eval())
                    biases_9.append(layer2_biases.eval())
                    biases_9.append(layer3_biases.eval())
                    biases_9.append(layer4_biases.eval())
                    biases_9.append(layer5_biases.eval())
                    biases_9.append(layer6_biases.eval())
                    biases_9.append(layer7_biases.eval())
                    biases_9.append(layer8_biases.eval())
                    biases_9.append(layer9_biases.eval())
                    biases_9.append(output_biases.eval())
                    
                    data = [weights_9, biases_9]
                    pickle.dump(data, open(pickle_path.format(stop_auc), "wb" ) )
                    print('Finished!')
        
        
            except:
                print('Fehler!')
                pass

print('Alles abgearbeitet!')