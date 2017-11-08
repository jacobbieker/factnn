from sklearn.metrics import roc_auc_score
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import h5py
import os


batch_size = 10000

path = '/fhgfs/users/jbehnken/crap/'
file = sorted(os.listdir(path))[-1]
path_loading = os.path.join(path, file)
print('Loading-File:', file)

load_weights, load_biases = pickle.load(open(path_loading, 'rb'))


def batchYielder(name):
    file_path = '/fhgfs/users/jbehnken/make_Data/MC_preprocessed_images.h5'
    #file_path = '/fhgfs/users/jbehnken/make_Data/MC_diffuse_preprocessed_images.h5'
    #file_path = '/fhgfs/users/jbehnken/make_Data/MC_diffuse_flat_preprocessed_images.h5'
    with h5py.File(file_path, 'r') as hdf:
        items = list(hdf.items())[0][1].shape[0]
        i = 0

        while (i+1)*batch_size < items/1: # 160 factor to not process everything
            images = np.array(hdf[name][ i*batch_size:(i+1)*batch_size ])

            i += 1
            print(i)
            if len(images)==batch_size:
                yield images
            
            
            
gpu_config = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.5)
session_conf = tf.ConfigProto(gpu_options=gpu_config, intra_op_parallelism_threads=18, inter_op_parallelism_threads=18)


with tf.Session(config=session_conf) as sess:
    graph = tf.get_default_graph()
    
    init_w_1 = tf.constant(load_weights[0])
    conv2d_1_weights = tf.get_variable('W_1', initializer=init_w_1)
    init_b_1 = tf.constant(load_biases[0])
    conv2d_1_biases = tf.get_variable('B_1', initializer=init_b_1)
    
    init_w_2 = tf.constant(load_weights[1])
    conv2d_2_weights = tf.get_variable('W_2', initializer=init_w_2)
    init_b_2 = tf.constant(load_biases[1])
    conv2d_2_biases = tf.get_variable('B_2', initializer=init_b_2)
    
    init_w_3 = tf.constant(load_weights[2])
    conv2d_3_weights = tf.get_variable('W_3', initializer=init_w_3)
    init_b_3 = tf.constant(load_biases[2])
    conv2d_3_biases = tf.get_variable('B_3', initializer=init_b_3)
    
    init_w_4 = tf.constant(load_weights[3])
    conv2d_4_weights = tf.get_variable('W_4', initializer=init_w_4)
    init_b_4 = tf.constant(load_biases[3])
    conv2d_4_biases = tf.get_variable('B_4', initializer=init_b_4)
    
    init_w_5 = tf.constant(load_weights[4])
    conv2d_5_weights = tf.get_variable('W_5', initializer=init_w_5)
    init_b_5 = tf.constant(load_biases[4])
    conv2d_5_biases = tf.get_variable('B_5', initializer=init_b_5)
    
    init_w_6 = tf.constant(load_weights[5])
    conv2d_6_weights = tf.get_variable('W_6', initializer=init_w_6)
    init_b_6 = tf.constant(load_biases[5])
    conv2d_6_biases = tf.get_variable('B_6', initializer=init_b_6)
    
    init_w_7 = tf.constant(load_weights[6])
    fc_1_weights = tf.get_variable('W_7', initializer=init_w_7)
    init_b_7 = tf.constant(load_biases[6])
    fc_1_biases = tf.get_variable('B_7', initializer=init_b_7)
    
    init_w_8 = tf.constant(load_weights[7])
    fc_2_weights = tf.get_variable('W_8', initializer=init_w_8)
    init_b_8 = tf.constant(load_biases[7])
    fc_2_biases = tf.get_variable('B_8', initializer=init_b_8)
    
    init_w_9 = tf.constant(load_weights[8])
    fc_3_weights = tf.get_variable('W_9', initializer=init_w_9)
    init_b_9 = tf.constant(load_biases[8])
    fc_3_biases = tf.get_variable('B_9', initializer=init_b_9)
    
    init_w_10 = tf.constant(load_weights[9])
    fc_4_weights = tf.get_variable('W_10', initializer=init_w_10)
    init_b_10 = tf.constant(load_biases[9])
    fc_4_biases = tf.get_variable('B_10', initializer=init_b_10)
    

    
    #tf_prediction_dataset = tf.constant(images, name='prediction_data')
    tf_prediction_dataset = tf.placeholder(tf.float32, shape=(batch_size, 46, 45, 1), name='training_data')
    sess.run(tf.global_variables_initializer())
    
    g_preds_1 = []
    g_preds_2 = []
    h_preds_1 = []
    h_preds_2 = []
    #sess.run(tf.global_variables_initializer())

    for images in batchYielder('Gamma'):
        feed_dict = {tf_prediction_dataset : images}
        
        with tf.name_scope('prediction_ccccccffff'):
            pool_1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(tf_prediction_dataset, conv2d_1_weights, [1, 1, 1, 1], padding='SAME') + conv2d_1_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_2 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_1, conv2d_2_weights, [1, 1, 1, 1], padding='SAME')  + conv2d_2_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_3 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_2, conv2d_3_weights, [1, 1, 1, 1], padding='SAME')  + conv2d_3_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_4 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_3, conv2d_4_weights, [1, 1, 1, 1], padding='SAME')  + conv2d_4_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_5 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_4, conv2d_5_weights, [1, 1, 1, 1], padding='SAME')  + conv2d_5_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_6 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_5, conv2d_6_weights, [1, 1, 1, 1], padding='SAME')  + conv2d_6_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            shape = pool_6.get_shape().as_list()
            reshape = tf.reshape(pool_6, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, fc_1_weights) + fc_1_biases)
            hidden = tf.nn.relu(tf.matmul(hidden, fc_2_weights) + fc_2_biases)
            hidden = tf.nn.relu(tf.matmul(hidden, fc_3_weights) + fc_3_biases)
            prediction_ccccccffff = tf.nn.softmax(tf.matmul(hidden, fc_4_weights) + fc_4_biases)
        pred = sess.run(prediction_ccccccffff, feed_dict=feed_dict)
       
        g_preds_1.extend(pred[:,0])
        g_preds_2.extend(pred[:,1])
        
        
        
        
    for images in batchYielder('Hadron'):
        feed_dict = {tf_prediction_dataset : images}

        with tf.name_scope('prediction_ccccccffff'):
            pool_1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(tf_prediction_dataset, conv2d_1_weights, [1, 1, 1, 1], padding='SAME') + conv2d_1_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_2 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_1, conv2d_2_weights, [1, 1, 1, 1], padding='SAME')  + conv2d_2_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_3 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_2, conv2d_3_weights, [1, 1, 1, 1], padding='SAME')  + conv2d_3_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_4 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_3, conv2d_4_weights, [1, 1, 1, 1], padding='SAME')  + conv2d_4_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_5 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_4, conv2d_5_weights, [1, 1, 1, 1], padding='SAME')  + conv2d_5_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_6 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_5, conv2d_6_weights, [1, 1, 1, 1], padding='SAME')  + conv2d_6_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            shape = pool_6.get_shape().as_list()
            reshape = tf.reshape(pool_6, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, fc_1_weights) + fc_1_biases)
            hidden = tf.nn.relu(tf.matmul(hidden, fc_2_weights) + fc_2_biases)
            hidden = tf.nn.relu(tf.matmul(hidden, fc_3_weights) + fc_3_biases)
            prediction_ccccccffff = tf.nn.softmax(tf.matmul(hidden, fc_4_weights) + fc_4_biases)
        pred = sess.run(prediction_ccccccffff, feed_dict=feed_dict)

        h_preds_1.extend(pred[:,0])
        h_preds_2.extend(pred[:,1])
        
        
        
g_data = list(zip(g_preds_1, g_preds_2))
h_data = list(zip(h_preds_1, h_preds_2))
g_df = pd.DataFrame(g_data, columns=['Hadron', 'Gamma'])
h_df = pd.DataFrame(h_data, columns=['Hadron', 'Gamma'])
#df.to_csv(prediction_save_path, index=False)

print('Gamma-Events:', len(g_data))
print('Hadron-Events:', len(h_data))



g_df['Real']=1
h_df['Real']=0

df = pd.concat([g_df, h_df])

stop_auc = roc_auc_score(df['Real'].values, df['Gamma'].values)
print('Auc:', stop_auc)



import matplotlib.pyplot as plt
path_build = '/home/jbehnken/07_make_FACT/build/'

plt.style.use('ggplot')
bins = np.arange(0,1.01,0.01)
ax = h_df.hist(['Gamma'], bins=bins, alpha=0.75, color=(238/255, 129/255, 10/255), figsize=(5, 3))
g_df.hist(['Gamma'], bins=bins, ax=ax, alpha=0.75, color=(118/255, 157/255, 6/255))
plt.yscale('log')
plt.legend(['Hadron', 'Gamma'], loc='lower center')
plt.title('Prediction of simulated events with AUC {:.3f}'.format(stop_auc))
plt.xlabel('Gamma ray probability')
plt.ylabel('Event count')
plt.tight_layout()
plt.savefig(path_build+'CNN_MC_Evaluation.pdf')
#plt.show()


