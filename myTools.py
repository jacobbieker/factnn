import tensorflow as tf
import numpy as np
import h5py

gpu_config = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.9)
session_conf = tf.ConfigProto(gpu_options=gpu_config, intra_op_parallelism_threads=9, inter_op_parallelism_threads=9)


with tf.Session(config=session_conf) as sess:
    saver = tf.train.import_meta_graph('CNN_Test_Model_3.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    graph = tf.get_default_graph()
    layer1_weights = graph.get_tensor_by_name("conv2d_1/W_1:0")
    layer1_biases = graph.get_tensor_by_name("conv2d_1/B_1:0")

    layer2_weights = graph.get_tensor_by_name("conv2d_2/W_2:0")
    layer2_biases = graph.get_tensor_by_name("conv2d_2/B_2:0")

    layer3_weights = graph.get_tensor_by_name("fc_1/W_3:0")
    layer3_biases = graph.get_tensor_by_name("fc_1/B_3:0")

    layer4_weights = graph.get_tensor_by_name("fc_2/W_4:0")
    layer4_biases = graph.get_tensor_by_name("fc_2/B_4:0")


    print("Model restored.")
