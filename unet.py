from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Model
from keras import backend as K
import yaml
import h5py


# Courteous of Joesph Yaconelli for the original implementation, modified by Jacob Bieker

try:
    with open("/run/media/jacob/SSD/Development/thesis/envs.yaml", 'r') as yaml_file:
        architecture = yaml.load(yaml_file)['arch']
except:
    architecture = 'intel'

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb1'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/thesis'

NUM_ALGOS = 1
INPUT_DIM_0 = 48
INPUT_DIM_1 = 48


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
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])

#from keras.datasets import mnist
import numpy as np

#(x_train, _), (x_test, _) = mnist.load_data()

# Load HDF5 data here
# Split into a test of images with 5000 events with same source position, etc. and labels of same image, and source position
# Maybe get the real source pos from Source database? and put it into SkyCoord pixels?
# But see if it can

# Validation one do it both with same source pos, if enough, and different source pos

# Print test images from source pos
np.random.seed(0)

with h5py.File(base_dir + "/FACTSources/crab_1314_std_analysis_v1.0.0_preprocessed_source.hdf5_55") as f:
    # Get some truth data for now, just use Crab images
    items = list(f.items())[0][1].shape[0]
    # Build up images with same source
    total_num = 400000
    source_one_images = []
    source_pos_one = []
    tmp_arr = np.zeros((46,45,1))
    k = 1
    source_truth = f['Source_Position'][0]
    for i in range(0, items):
        if not np.array_equal(f['Source_Position'][i], source_truth) and f['Trigger'][i] == 4:
            source_truth_two = f['Source_Position'][i]
    for i in range(0, items):
        if np.array_equal(f['Source_Position'][i], source_truth) and f['Trigger'][i] == 4:
            # arrays are the same, add to source images and ones
            # Randomly flip the array twice to augment training

            if (k % 500) != 0:
                # Add to temp image
                tmp_arr += f['Image'][i]
                k += 1
            else:
                # Hit the 5000 cap I need
                print("5000 Hit")
                # REsize correctly
                tmp_arr = np.c_[tmp_arr.reshape((46,45)), np.zeros((46,3))]
                tmp_arr = np.r_[tmp_arr, np.zeros((2,48))]
                tmp_arr = tmp_arr.reshape((48,48,1))
                #tmp_arr.resize((48,48,1))
                source_one_images.append(tmp_arr)
                source_arr = f['Source_Position'][i]
                source_arr = np.c_[source_arr.reshape((46,45)), np.zeros((46,3))]
                source_arr = np.r_[source_arr, np.zeros((2,48))]
                source_arr = source_arr.reshape((48,48,1))
                #source_arr.resize((48,48,1), refcheck=False)
                source_pos_one.append(source_arr)
                tmp_arr = np.zeros((46,45,1))
                k += 1


            ran_int = np.random.randint(0,3)
            ''''
            if ran_int < 5:
                image_arr = f['Image'][i]
                image_arr.resize((48,48,1), refcheck=False)
                source_one_images.append(image_arr)
                source_arr = f['Source_Position'][i]
                source_arr.resize((48,48,1), refcheck=False)
                source_pos_one.append(source_arr)
            else:
                print("Flipping")
                image_arr = f['Image'][i]
                # Flip twice
                image_arr = np.fliplr(image_arr)
                image_arr = np.flipud(image_arr)
                print(image_arr.shape)
                image_arr.resize((48,48,1), refcheck=False)
                source_one_images.append(image_arr)
                source_arr = f['Source_Position'][i]
                # Flip twice
                source_arr = np.fliplr(image_arr)
                source_arr = np.flipud(image_arr)
                source_arr.resize((48,48,1), refcheck=False)
                source_pos_one.append(source_arr)
            '''
        #elif np.array_equal(f['Source_Position'][i], source_truth_two) and f['Trigger'][i] == 4:
        #    image_arr = f['Image'][i]
        #    image_arr.resize((48,48,1))
        #    source_two_images.append(image_arr)
        #    source_arr = f['Source_Position'][i]
        #    source_arr.resize((48,48,1))
        #    source_pos_two.append(source_arr)

    x_train = np.array(source_one_images)
    print(x_train.shape)
    #x_test = source_two_images
    x_train_source = np.array(source_pos_one)
    print(x_train_source.shape)
    #x_test_source = source_pos_two

import matplotlib.pyplot as plt

with h5py.File("/run/media/jacob/WDRed8Tb2/FACTSources/FromTalapas/mrk501_2014_std_analysis_v1.0.0_preprocessed_source.hdf5_195") as f:
    # Get some truth data for now, just use Crab images
    items = list(f.items())[0][1].shape[0]
    # Build up images with same source
    source_two_images = []
    source_pos_two = []
    source_truth = f['Source_Position'][0]
    tmp_arr = np.zeros((46,45,1))
    tmp_arr2 = np.zeros((46,45,1))
    source_arr2 = np.zeros((46,45,1))
    k = 1
    for i in range(0, items):
        if not np.array_equal(f['Source_Position'][i], source_truth) and f['Trigger'][i] == 4:
            source_truth_two = f['Source_Position'][i]
    for i in range(0, items):
        if np.array_equal(f['Source_Position'][i], source_truth) and f['Trigger'][i] == 4:
            # arrays are the same, add to source images and ones
            # Randomly flip the array twice to augment training

            ran_int = np.random.randint(0,3)

            if (k % 500) != 0:
                # Add to temp image
                tmp_arr += f['Image'][i]
                tmp_arr2 += f['Image'][i]
                k += 1
            else:
                # Hit the 5000 cap I need
                #plt.imshow(tmp_arr.reshape(46, 45))
                #plt.show()
                tmp_arr = np.c_[tmp_arr.reshape((46,45)), np.zeros((46,3))]
                tmp_arr = np.r_[tmp_arr, np.zeros((2,48))]
                tmp_arr = tmp_arr.reshape((48,48,1))
                #plt.imshow(tmp_arr.reshape(48, 48))
                #plt.show()
                source_two_images.append(tmp_arr)
                source_arr = f['Source_Position'][i]
                source_arr = np.c_[source_arr.reshape((46,45)), np.zeros((46,3))]
                source_arr = np.r_[source_arr, np.zeros((2,48))]
                source_arr = source_arr.reshape((48,48,1))
                source_pos_two.append(source_arr)
                tmp_arr = np.zeros((46,45,1))
                k += 1

    #plt.imshow(tmp_arr2.reshape((46,45)))
    #plt.title("All events here")
    #plt.show()

    #source_arr2 = f['Source_Position'][0]

    #plt.imshow(source_arr2.reshape((46,45)))
    #plt.title("Source of all events")
    #plt.show()

    '''
    if ran_int < 5:
        image_arr = f['Image'][i]
        image_arr.resize((48,48,1), refcheck=False)
        source_two_images.append(image_arr)
        source_arr = f['Source_Position'][i]
        source_arr.resize((48,48,1), refcheck=False)
        source_pos_two.append(source_arr)
    else:
        print("Flipping")
        image_arr = f['Image'][i]
        # Flip twice
        image_arr = np.fliplr(image_arr)
        image_arr = np.flipud(image_arr)
        print(image_arr.shape)
        image_arr.resize((48,48,1), refcheck=False)
        source_two_images.append(image_arr)
        source_arr = f['Source_Position'][i]
        # Flip twice
        source_arr = np.fliplr(image_arr)
        source_arr = np.flipud(image_arr)
        source_arr.resize((48,48,1), refcheck=False)
        source_pos_two.append(source_arr)
    '''
    # now get number of 500 event things, or 5000 event things
    x_test = np.array(source_two_images)
    print(x_test.shape)
    #x_test = source_two_images
    x_test_source = np.array(source_pos_two)
    print(x_test_source.shape)
    print("Finished getting data")

x_train = np.reshape(x_train, (-1, 48, 48, 1))
x_train_source = np.reshape(x_train_source, (-1, 48, 48, 1))
x_test = np.reshape(x_test, (-1, 48, 48, 1))
x_test_source = np.reshape(x_test_source, (-1, 48, 48, 1))

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
    validation_split=0.2,
    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

import matplotlib.pyplot as plt

autoencoder.save(filepath="unet_test_500.hdf5")
decoded_imgs = autoencoder.predict(x_test)

# number of images to display
n = 10
plt.figure(figsize=(20, 4))

for i in range(1, n+1):
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test_source[i].reshape(48, 48))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(48, 48))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
