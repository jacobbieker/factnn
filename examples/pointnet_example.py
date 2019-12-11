__copyright__ = "Copyright (C) 2019 HP Development Company, L.P."
# SPDX-License-Identifier: MIT
import keras
from keras.optimizers import adam

from keras import backend as K
from keras.layers import Input, Dropout, Flatten, Dense, MaxPooling2D, Dot, Lambda, \
    Reshape, BatchNormalization, Activation, Conv1D, AveragePooling2D
from keras.initializers import Constant
from keras.models import Model
from keras.regularizers import Regularizer
import keras.utils as keras_utils

import numpy as np
from factnn.generator.keras.pointcloud_generator import PointCloudGenerator
from factnn.data.preprocess.pointcloud_preprocessor import PointCloudPreprocessor
from factnn.utils.cross_validate import fit_model


class OrthogonalRegularizer(Regularizer):
    """
    Considering that input is flattened square matrix X, regularizer tries to ensure that matrix X
    is orthogonal, i.e. ||X*X^T - I|| = 0. L1 and L2 penalties can be applied to it
    """

    def __init__(self, l1=0.0, l2=0.0):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)

    def __call__(self, x):
        size = int(np.sqrt(x.shape[1].value))
        assert (size * size == x.shape[1].value)
        x = K.reshape(x, (-1, size, size))
        xxt = K.batch_dot(x, x, axes=(2, 2))
        regularization = 0.0
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(xxt - K.eye(size)))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(xxt - K.eye(size)))

        return regularization

    def get_config(self):
        return {'l1': float(self.l1), 'l2': float(self.l2)}


def orthogonal(l1=0.0, l2=0.0):
    """
    Functional wrapper for OrthogonalRegularizer.
    :param l1: l1 penalty
    :param l2: l2 penalty
    :return: Orthogonal regularizer to append to a loss function
    """
    return OrthogonalRegularizer(l1=l1, l2=l2)


def dense_bn(x, units, use_bias=True, scope=None, activation=None):
    """
    Utility function to apply Dense + Batch Normalization.
    """
    with K.name_scope(scope):
        x = Dense(units=units, use_bias=use_bias)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation(activation)(x)
    return x


def conv1d_bn(x, num_filters, kernel_size, padding='same', strides=1,
              use_bias=False, scope=None, activation='relu'):
    """
    Utility function to apply Convolution + Batch Normalization.
    """
    with K.name_scope(scope):
        input_shape = x.get_shape().as_list()[-2:]
        x = Conv1D(num_filters, kernel_size, strides=strides, padding=padding,
                   use_bias=use_bias, input_shape=input_shape)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation(activation)(x)
    return x


def transform_net(inputs, scope=None, regularize=False):
    """
    Generates an orthogonal transformation tensor for the input data
    :param inputs: tensor with input image (either BxNxK or BxNx1xK)
    :param scope: name of the grouping scope
    :param regularize: enforce orthogonality constraint
    :return: BxKxK tensor of the transformation
    """
    with K.name_scope(scope):
        input_shape = inputs.get_shape().as_list()
        k = input_shape[-1]
        num_points = input_shape[-2]

        net = conv1d_bn(inputs, num_filters=128, kernel_size=1, padding='valid',
                        use_bias=True, scope='tconv1')
        net = conv1d_bn(net, num_filters=256, kernel_size=1, padding='valid',
                        use_bias=True, scope='tconv2')
        net = conv1d_bn(net, num_filters=2048, kernel_size=1, padding='valid',
                        use_bias=True, scope='tconv3')

        #  Done in 2D since 1D is painfully slow
        net = MaxPooling2D(pool_size=(num_points, 1), padding='valid')(Lambda(K.expand_dims)(net))
        net = Flatten()(net)

        net = dense_bn(net, units=1024, scope='tfc1', activation='relu')
        net = dense_bn(net, units=512, scope='tfc2', activation='relu')

        transform = Dense(units=k * k,
                          kernel_initializer='zeros', bias_initializer=Constant(np.eye(k).flatten()),
                          activity_regularizer=orthogonal(l2=0.001) if regularize else None)(net)
        transform = Reshape((k, k))(transform)

    return transform


def pointnet_base(inputs, use_tnet=True):
    """
    Convolutional portion of pointnet, common across different tasks (classification, segmentation, etc)
    :param inputs: Input tensor with the point cloud shape (BxNxK)
    :param use_tnet: whether to use the transformation subnets or not.
    :return: tensor layer for CONV5 activations
    """

    # Obtain spatial point transform from inputs and convert inputs
    if use_tnet:
        ptransform = transform_net(inputs, scope='transform_net1', regularize=False)
        point_cloud_transformed = Dot(axes=(2, 1))([inputs, ptransform])

    # First block of convolutions
    net = conv1d_bn(point_cloud_transformed if use_tnet else inputs, num_filters=128, kernel_size=1, padding='valid',
                    use_bias=True, scope='conv1')
    net = conv1d_bn(net, num_filters=128, kernel_size=1, padding='valid',
                    use_bias=True, scope='conv2')

    # Obtain feature transform and apply it to the network
    if use_tnet:
        ftransform = transform_net(net, scope='transform_net2', regularize=True)
        net_transformed = Dot(axes=(2, 1))([net, ftransform])

    # Second block of convolutions
    net = conv1d_bn(net_transformed if use_tnet else net, num_filters=64, kernel_size=1, padding='valid',
                    use_bias=True, scope='conv3')
    net = conv1d_bn(net, num_filters=256, kernel_size=1, padding='valid',
                    use_bias=True, scope='conv4')
    net = conv1d_bn(net, num_filters=2048, kernel_size=1, padding='valid',
                    use_bias=True, scope='conv5')

    return net


def pointnet_cls(include_top=True, weights=None, input_shape=(2048, 3),
                 pooling=None, classes=40, activation=None, use_tnet=True):
    """
    PointNet model for object classification
    :param include_top: whether to include the stack of fully connected layers
    :param weights: one of `None` (random initialization),
                    'modelnet' (pre-training on ModelNet),
                    or the path to the weights file to be loaded.
    :param input_shape: shape of the input point clouds (NxK)
    :param pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 2D tensor output of the last convolutional block (Nx1024).
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 1D tensor of size 1024.
            - `max` means that global max pooling will
                be applied.
    :param classes: number of classes in the classification problem; if dict, construct multiple disjoint top layers
    :param activation: activation of the last layer (default None).
    :param use_tnet: whether to use the transformation subnets or not.
    :return: Keras model of the classification network
    """

    assert K.image_data_format() == 'channels_last'
    num_point = input_shape[0]

    # Generate input tensor and get base network
    inputs = Input(input_shape, name='Input_cloud')
    net = pointnet_base(inputs, use_tnet)

    # Top layers
    if include_top:
        # Symmetric function: max pooling
        # Done in 2D since 1D is painfully slow
        net = MaxPooling2D(pool_size=(num_point, 1), padding='valid', name='maxpool')(Lambda(K.expand_dims)(net))
        net = Flatten()(net)
        if isinstance(classes, dict):
            # Disjoint stacks of fc layers, one per value in dict
            net = [dense_bn(net, units=1024, scope=r + '_fc1', activation='relu') for r in classes]
            net = [Dropout(0.3, name=r + '_dp1')(n) for r, n in zip(classes, net)]
            net = [dense_bn(n, units=512, scope=r + '_fc2', activation='relu') for r, n in zip(classes, net)]
            net = [Dropout(0.3, name=r + '_dp2')(n) for r, n in zip(classes, net)]
            net = [Dense(units=classes[r], activation=activation, name=r)(n) for r, n in zip(classes, net)]
        else:
            # Fully connected layers for a single classification task
            net = dense_bn(net, units=1024, scope='fc1', activation='relu')
            net = Dropout(0.3, name='dp1')(net)
            net = dense_bn(net, units=512, scope='fc2', activation='relu')
            net = Dropout(0.3, name='dp2')(net)
            net = Dense(units=classes, name='fc3', activation=activation)(net)
    else:
        if pooling == 'avg':
            net = MaxPooling2D(pool_size=(num_point, 1), padding='valid', name='maxpool')(Lambda(K.expand_dims)(net))
        elif pooling == 'max':
            net = AveragePooling2D(pool_size=(num_point, 1), padding='valid', name='avgpool')(
                Lambda(K.expand_dims)(net))

    model = Model(inputs, net, name='pointnet_cls')

    # Load weights.
    if weights == 'modelnet':
        pass
        if K.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights, by_name=True)

    return model


directory = "/home/jacob/"
gamma_dir = [directory + "gammaFeature/core5/"]
proton_dir = [directory + "protonFeature/core5/"]
import os

paths = []
for source_dir in gamma_dir:
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            paths.append(os.path.join(root, file))
paths = paths#[:5000]
train_paths = paths[:int(.6 * len(paths))]
val_paths = paths[int(.6 * len(paths)):int(.8 * len(paths))]
test_paths = paths[int(.8 * len(paths)):]

proton_paths = []
for source_dir in proton_dir:
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            proton_paths.append(os.path.join(root, file))
proton_paths = proton_paths#[:5000]
train_p_paths = paths[:int(.6 * len(proton_paths))]
val_p_paths = paths[int(.6 * len(proton_paths)):int(.8 * len(proton_paths))]
test_p_paths = paths[int(.8 * len(proton_paths)):]

gamma_configuration = {
    'paths': train_paths,
    'rebin_size': 5,
    'shape': (0,160)
}
proton_configuration = {
    'paths': train_p_paths,
    'rebin_size': 5,
    'shape': (0,160)
}

gamma_train_preprocessor = PointCloudPreprocessor(config=gamma_configuration)
gamma_configuration["paths"] = val_paths
gamma_val_preprocessor = PointCloudPreprocessor(config=gamma_configuration)
gamma_configuration["paths"] = test_paths
gamma_test_preprocessor = PointCloudPreprocessor(config=gamma_configuration)

proton_train_preprocessor = PointCloudPreprocessor(config=proton_configuration)
proton_configuration["paths"] = val_p_paths
proton_val_preprocessor = PointCloudPreprocessor(config=proton_configuration)
proton_configuration["paths"] = test_p_paths
proton_test_preprocessor = PointCloudPreprocessor(config=proton_configuration)

final_points=1024

train = PointCloudGenerator(train_paths, batch_size=16, preprocessor=gamma_train_preprocessor,
                            proton_paths=train_p_paths,
                            proton_preprocessor=proton_train_preprocessor,
                            slices=(0, 160),
                            final_points=final_points,
                            replacement=False,
                            augment=True,
                            training_type="Separation",
                            truncate=False,
                            return_features=False,
                            rotate=True, )

val = PointCloudGenerator(val_paths, batch_size=16, preprocessor=gamma_val_preprocessor,
                          proton_paths=val_p_paths,
                          proton_preprocessor=proton_val_preprocessor,
                          slices=(0, 160),
                          final_points=final_points,
                          replacement=False,
                          augment=False,
                          training_type="Separation",
                          truncate=False,
                          return_features=False,
                          rotate=True, )

model = pointnet_cls(input_shape=(final_points, 3), classes=2, use_tnet=True, activation='softmax')
loss = 'binary_crossentropy'
#loss = "mean_squared_error"
metric = ['accuracy']
monitor = 'val_loss'

# tensorboard and weights saving callbacks
callbacks = list()
callbacks.append(keras.callbacks.TensorBoard(log_dir="./", histogram_freq=0, write_graph=True))
callbacks.append(
    keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=5, verbose=1, min_lr=1e-10))
callbacks.append(keras.callbacks.EarlyStopping(monitor=monitor, patience=20))
callbacks.append(keras.callbacks.TerminateOnNaN())

#callbacks.append(keras.callbacks.ModelCheckpoint(weights_path, monitor=monitor, verbose=0, save_best_only=True,
#
optimizer = adam(lr=0.001)
model.compile(loss=loss, optimizer=optimizer, metrics=metric)

model.summary()

model.fit_generator(
    generator=train,
    epochs=500,
    verbose=2,
    validation_data=val,
    callbacks=callbacks,
    use_multiprocessing=True,
    workers=12,
    max_queue_size=400,
)
