import numpy as np
from sklearn.utils import shuffle


def image_augmenter(images):
    """
    Augment images by rotating and flipping input images randomly
    Does this on the 2nd and 3rd axis of each 4D image stack
    :param images: Numpy list of images in (batch_size, timeslice, x, y, channels) format
    :return: Numpy array of randomly flipped and rotated 3D images, in same order
    """
    new_images = []
    for image in images:
        vert_val = np.random.rand()
        if vert_val < 0.5:
            # Flip the image vertically
            image = np.flip(image, 1)
        horz_val = np.random.rand()
        if horz_val < 0.5:
            # Flip horizontally
            image = np.flip(image, 2)
        rot_val = np.random.rand()
        if rot_val < 0.3:
            # Rotate 90 degrees
            image = np.rot90(image, 1, axes=(1, 2))
        elif rot_val > 0.7:
            # Rotate 270 degrees
            image = np.rot90(image, 3, axes=(1, 2))

        new_images.append(image)
    images = np.asarray(new_images)
    return images


def get_random_hdf5_chunk(start, stop, size, time_slice, total_slices, training_data, labels=None, proton_data=None,
                          type_training=None, augment=True, swap=True):
    '''
    Gets a random part of the HDF5 database within start and stop endpoints
    This is to help with shuffling data, as currently all the ones come and go in the same
    order
    Does not guarantee that a given event will be used though, unlike before
    Recommended to alternate this with the current one to make sure network has full coverage

    :param labels:
    :param type_training:
    :param proton_data:
    :param training_data:
    :param start:
    :param stop:
    :param size:
    :param time_slice:
    :param total_slices:
    :return:
    '''

    # Get all possible starting positions given the end point and number of events

    last_possible_start = stop - size

    # Get random starting position
    start_pos = np.random.randint(start, last_possible_start)

    batch_images = training_data[start_pos:int(start_pos + size), time_slice - total_slices:time_slice, ::]
    if augment:
        batch_images = image_augmenter(batch_images)
    if type_training == "Separation":
        proton_images = proton_data[start_pos:int(start_pos + size), time_slice - total_slices:time_slice, ::]
        if augment:
            proton_images = image_augmenter(proton_images)
        labels = np.array([True] * (len(batch_images)) + [False] * len(proton_images))
        batch_image_label = (np.arange(2) == labels[:, None]).astype(np.float32)
        batch_images = np.concatenate([batch_images, proton_images], axis=0)

        if swap:
            batch_images, batch_image_label = shuffle(batch_images, batch_image_label)
        return batch_images, batch_image_label
    else:
        labels = labels[start_pos:int(start_pos + size)]
        batch_image_label = labels
        if swap:
            batch_images, batch_image_label = shuffle(batch_images, batch_image_label)
        return batch_images, batch_image_label


def get_completely_random_hdf5(start, stop, size, time_slice, total_slices, training_data, labels=None,
                               proton_data=None, type_training=None, augment=True, swap=True):
    '''
    Gets a random part of the HDF5 database within start and stop endpoints
    This is to help with shuffling data, as currently all the ones come and go in the same
    order
    Does not guarantee that a given event will be used though, unlike before
    Recommended to alternate this with the current one to make sure network has full coverage
    This variant obtians a list of random points, so it will be lower than the other options, but should be better for
    training

    :param labels:
    :param type_training:
    :param proton_data:
    :param training_data:
    :param start:
    :param stop:
    :param size:
    :param time_slice:
    :param total_slices:
    :return:
    '''

    # Get random positions within the start and stop sizes
    positions = np.random.randint(start, stop, size=size)

    batch_images = training_data[positions, time_slice - total_slices:time_slice, ::]
    if augment:
        batch_images = image_augmenter(batch_images)
    if type_training == "Separation":
        proton_images = proton_data[positions, time_slice - total_slices:time_slice, ::]
        if augment:
            proton_images = image_augmenter(proton_images)
        labels = np.array([True] * (len(batch_images)) + [False] * len(proton_images))
        batch_image_label = (np.arange(2) == labels[:, None]).astype(np.float32)
        batch_images = np.concatenate([batch_images, proton_images], axis=0)
        if swap:
            batch_images, batch_image_label = shuffle(batch_images, batch_image_label)
        return batch_images, batch_image_label
    else:
        labels = labels[positions]
        batch_image_label = labels
        if swap:
            batch_images, batch_image_label = shuffle(batch_images, batch_image_label)
        return batch_images, batch_image_label


def get_random_from_list(indicies, size, time_slice, total_slices, training_data, labels=None,
                               proton_data=None, type_training=None, augment=True, swap=True):
    '''
    Gets a random part of the HDF5 database within start and stop endpoints
    This is to help with shuffling data, as currently all the ones come and go in the same
    order
    Does not guarantee that a given event will be used though, unlike before
    Recommended to alternate this with the current one to make sure network has full coverage
    This variant obtians a list of random points, so it will be lower than the other options, but should be better for
    training

    :param labels:
    :param type_training:
    :param proton_data:
    :param training_data:
    :param start:
    :param stop:
    :param size:
    :param time_slice:
    :param total_slices:
    :return:
    '''

    # Get random positions within the start and stop sizes
    positions = np.random.choice(indicies, size=size, replace=False)

    batch_images = training_data[positions, time_slice - total_slices:time_slice, ::]
    if augment:
        batch_images = image_augmenter(batch_images)
    if type_training == "Separation":
        proton_images = proton_data[positions, time_slice - total_slices:time_slice, ::]
        if augment:
            proton_images = image_augmenter(proton_images)
        labels = np.array([True] * (len(batch_images)) + [False] * len(proton_images))
        batch_image_label = (np.arange(2) == labels[:, None]).astype(np.float32)
        batch_images = np.concatenate([batch_images, proton_images], axis=0)
        if swap:
            batch_images, batch_image_label = shuffle(batch_images, batch_image_label)
        return batch_images, batch_image_label
    else:
        labels = labels[positions]
        batch_image_label = labels
        if swap:
            batch_images, batch_image_label = shuffle(batch_images, batch_image_label)
        return batch_images, batch_image_label
