import keras
from keras.optimizers import adam
import sys

sys.path.insert(0, './')

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

from factnn.generator.keras.pointcloud_generator import PointCloudGenerator
from factnn.data.preprocess.pointcloud_preprocessor import PointCloudPreprocessor
from factnn.utils.cross_validate import fit_model

from .pnet2_layers.layers import Pointnet_SA, Pointnet_SA_MSG, Pointnet_FP

directory = "/home/jacob/"
gamma_dir = [directory + "gammaFeature/core5/"]
proton_dir = [directory + "protonFeature/core5/"]
import os

paths = []
for source_dir in gamma_dir:
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            paths.append(os.path.join(root, file))
paths = paths  # [:5000]
train_paths = paths[:int(.6 * len(paths))]
val_paths = paths[int(.6 * len(paths)):int(.8 * len(paths))]
test_paths = paths[int(.8 * len(paths)):]

proton_paths = []
for source_dir in proton_dir:
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            proton_paths.append(os.path.join(root, file))
proton_paths = proton_paths  # [:5000]
train_p_paths = paths[:int(.6 * len(proton_paths))]
val_p_paths = paths[int(.6 * len(proton_paths)):int(.8 * len(proton_paths))]
test_p_paths = paths[int(.8 * len(proton_paths)):]

gamma_configuration = {
    'paths': train_paths,
    'rebin_size': 5,
    'shape': (0, 160)
}
proton_configuration = {
    'paths': train_p_paths,
    'rebin_size': 5,
    'shape': (0, 160)
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

final_points = 1024

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


class CLS_MSG_Model(Model):

    def __init__(self, batch_size, num_points, num_classes, bn=False, activation=tf.nn.leaky_relu):
        super(CLS_MSG_Model, self).__init__()

        self.activation = activation
        self.batch_size = batch_size
        self.num_points = num_points
        self.num_classes = num_classes
        self.bn = bn
        self.keep_prob = 0.4

        self.kernel_initializer = 'glorot_normal'
        self.kernel_regularizer = None

        self.init_network()

    def init_network(self):

        self.layer1 = Pointnet_SA_MSG(
            npoint=1024,
            radius_list=[0.1, 0.2, 0.4],
            nsample_list=[16, 32, 128],
            mlp=[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
            activation=self.activation,
            bn=self.bn
        )

        self.layer2 = Pointnet_SA_MSG(
            npoint=512,
            radius_list=[0.2, 0.4, 0.8],
            nsample_list=[32, 64, 128],
            mlp=[[64, 64, 128], [128, 128, 256], [128, 128, 256]],
            activation=self.activation,
            bn=self.bn
        )

        self.layer3 = Pointnet_SA(
            npoint=None,
            radius=None,
            nsample=None,
            mlp=[256, 512, 1024],
            group_all=True,
            activation=self.activation,
            bn=self.bn
        )

        self.dense1 = Dense(512, activation=self.activation)
        if self.bn: self.bn_fc1 = BatchNormalization()

        self.dropout1 = Dropout(self.keep_prob)

        self.dense2 = Dense(128, activation=self.activation)
        if self.bn: self.bn_fc2 = BatchNormalization()

        self.dropout2 = Dropout(self.keep_prob)

        self.dense3 = Dense(self.num_classes, activation=tf.nn.softmax)

    def call(self, input, training=True):

        xyz, points = self.layer1(input, None, training=training)
        xyz, points = self.layer2(xyz, points, training=training)
        xyz, points = self.layer3(xyz, points, training=training)

        net = tf.reshape(points, (self.batch_size, -1))

        net = self.dense1(net)
        if self.bn: net = self.bn_fc1(net, training=training)
        net = self.dropout1(net)

        net = self.dense2(net)
        if self.bn: net = self.bn_fc2(net, training=training)
        net = self.dropout2(net)

        pred = self.dense3(net)

        return pred


model = CLS_MSG_Model(16, final_points, 2, False)
loss = 'binary_crossentropy'
# loss = "mean_squared_error"
metric = ['accuracy']
monitor = 'val_loss'

# tensorboard and weights saving callbacks
callbacks = list()
callbacks.append(keras.callbacks.TensorBoard(log_dir="./", histogram_freq=0, write_graph=True))
callbacks.append(
    keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=5, verbose=1, min_lr=1e-10))
callbacks.append(keras.callbacks.EarlyStopping(monitor=monitor, patience=20))
callbacks.append(keras.callbacks.TerminateOnNaN())

# callbacks.append(keras.callbacks.ModelCheckpoint(weights_path, monitor=monitor, verbose=0, save_best_only=True,
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
