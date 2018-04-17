from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Model
from keras import backend as K
import yaml
import h5py


# Courteous of Joesph Yaconelli for the original implementation, modified by Jacob Bieker

with open("/run/media/jacob/SSD/Development/thesis/envs.yaml", 'r') as yaml_file:
    architecture = yaml.load(yaml_file)['arch']

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb1'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/thesis'

NUM_ALGOS = 1
INPUT_DIM_0 = 46
INPUT_DIM_1 = 45


input_img = Input(shape=(INPUT_DIM_0, INPUT_DIM_1, NUM_ALGOS))

conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
print("conv1 shape:",conv1.shape)

pool1 = MaxPooling2D((2, 2), padding='same')(conv1)
print("pool1 shape:",pool1.shape)

conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
print("conv2 shape:",conv2.shape)

pool2 = MaxPooling2D((2, 2), padding='same')(conv2)
print("pool2 shape:",pool2.shape)


conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
print("conv3 shape:",conv3.shape)

#pool3 = MaxPooling2D((2, 2), padding='same')(conv3)
#print("pool3 shape:",pool3.shape)

conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
print("conv4 shape:",conv4.shape)

up1 = Conv2D(32, (3, 3), activation='relu', padding='same')(UpSampling2D((2, 2))(conv4))
print("up1 shape:", up1.shape)
#x = Conv2D(32, (3, 3), activation='relu', padding='same')(UpSampling2D((2, 2))(up1))
#print("x shape:",x.shape)

merge1 = concatenate([conv2, up1], axis=3)
print("merge1 shape:",merge1.shape)
conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge1)

up2 = Conv2D(16, (3, 3), activation='relu', padding='same')(UpSampling2D((2, 2))(conv5))
merge2 = concatenate([conv1, up2], axis=3)
conv6 = Conv2D(16, (3, 3), activation='relu', padding='same')(merge2)

decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(conv6)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

# Load HDF5 data here
# Split into a test of images with 5000 events with same source position, etc. and labels of same image, and source position
# Maybe get the real source pos from Source database? and put it into SkyCoord pixels?
# But see if it can

# Validation one do it both with same source pos, if enough, and different source pos

# Print test images from source pos

with h5py.File(base_dir + "/FACTSources/crab_1314_std_analysis_v1.0.0_preprocessed_source.hdf5_34") as f:
    # Get some truth data for now, just use Crab images
    items = list(f.items())[0][1].shape[0]
    # Build up images with same source
    total_num = 400000
    source_one_images = []
    source_two_images = []
    source_pos_one = []
    source_pos_two = []
    source_truth = f['Source_Position'][0]
    for i in range(0, items):
        if not np.array_equal(f['Source_Position'][i], source_truth) and f['Trigger'][i] == 4:
            source_truth_two = f['Source_Position'][i]
    for i in range(0, items):
        if np.array_equal(f['Source_Position'][i], source_truth) and f['Trigger'][i] == 4:
            # arrays are the same, add to source images and ones
            source_one_images.append(f['Images'][i])
            source_pos_one.append(f['Source_Position'][i])
        elif np.array_equal(f['Source_Position'][i], source_truth_two) and f['Trigger'][i] == 4:
            source_two_images.append(f['Images'][i])
            source_pos_two.append(f['Source_Position'][i])

    x_train = source_one_images
    x_test = source_two_images
    x_train_source = source_pos_one
    x_test_source = source_pos_two

    # now get number of 500 event things, or 5000 event things

    print("Finished getting data")

x_train = np.reshape(x_train, (len(x_train), 46, 45, 1))
x_test = np.reshape(x_test, (len(x_test), 46, 45, 1))

#noise_factor = 0.5
#x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
#x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

#x_train_noisy = np.clip(x_train_noisy, 0., 1.)
#x_test_noisy = np.clip(x_test_noisy, 0., 1.)


from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train_source,
    epochs = 10,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test, x_test_source),
    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

import matplotlib.pyplot as plt

decoded_imgs = autoencoder.predict(x_test)

# number of images to display
n = 10
plt.figure(figsize=(20, 4))

for i in range(1, n+1):
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test_source[i].reshape(46, 45))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(46, 45))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
