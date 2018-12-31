import numpy as np
from sklearn.utils import shuffle
import h5py


def image_augmenter(images, as_channels=False):
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
            if not as_channels:
                image = np.flip(image, 1)
            else:
                image = np.flip(image, 0)
        horz_val = np.random.rand()
        if horz_val < 0.5:
            # Flip horizontally
            if not as_channels:
                image = np.flip(image, 2)
            else:
                image = np.flip(image, 1)
        rot_val = np.random.rand()
        if rot_val < 0.3:
            # Rotate 90 degrees
            if not as_channels:
                image = np.rot90(image, 1, axes=(1, 2))
            else:
                image = np.rot90(image, 1, axes=(0, 1))
        elif rot_val > 0.7:
            # Rotate 270 degrees
            if not as_channels:
                image = np.rot90(image, 3, axes=(1, 2))
            else:
                image = np.rot90(image, 3, axes=(0, 1))

        new_images.append(image)
    images = np.asarray(new_images)
    return images


def dual_image_augmenter(images, collapsed_images, as_channels=False):
    """
    Augment images by rotating and flipping input images randomly
    Does this on the 2nd and 3rd axis of each 4D image stack
    :param images: Numpy list of images in (batch_size, timeslice, x, y, channels) format
    :return: Numpy array of randomly flipped and rotated 3D images, in same order
    """
    new_images = []
    new_collapsed_images = []
    for index, image in enumerate(images):
        collapsed_image = collapsed_images[index]
        vert_val = np.random.rand()
        if vert_val < 0.5:
            # Flip the image vertically
            if not as_channels:
                image = np.flip(image, 1)
            else:
                image = np.flip(image, 0)
            collapsed_image = np.flip(collapsed_image,0)
        horz_val = np.random.rand()
        if horz_val < 0.5:
            # Flip horizontally
            if not as_channels:
                image = np.flip(image, 2)
            else:
                image = np.flip(image, 1)
            collapsed_image = np.flip(collapsed_image, 1)
        rot_val = np.random.rand()
        if rot_val < 0.3:
            # Rotate 90 degrees
            if not as_channels:
                image = np.rot90(image, 1, axes=(1, 2))
            else:
                image = np.rot90(image, 1, axes=(0, 1))
            collapsed_image = np.rot90(collapsed_image, 1, axes=(0,1))
        elif rot_val > 0.7:
            # Rotate 270 degrees
            if not as_channels:
                image = np.rot90(image, 3, axes=(1, 2))
            else:
                image = np.rot90(image, 3, axes=(0, 1))
            collapsed_image = np.rot90(collapsed_image, 3, axes=(0,1))
        new_images.append(image)
        new_collapsed_images.append(collapsed_image)
    collapsed_images = np.asarray(new_collapsed_images)
    images = np.asarray(new_images)
    return images, collapsed_images


def common_step(batch_images, positions=None, labels=None, proton_images=None, augment=True, swap=True, shape=None,
                as_channels=False, return_collapsed=False, return_features=False):

    # Get the correct index for the collapsed and feature data if used
    if return_features and return_collapsed:
        feature_index = 1
        collapsed_index = 2
    elif return_features and not return_collapsed:
        feature_index = 1
        collapsed_index = -99
    elif not return_features and return_collapsed:
        feature_index = -99
        collapsed_index = 1
    else:
        feature_index = -99
        collapsed_index = -99

    if augment:
        if return_collapsed:
            batch_images[0], batch_images[collapsed_index] = dual_image_augmenter(batch_images[0], batch_images[collapsed_index], as_channels)
        else:
            batch_images = image_augmenter(batch_images, as_channels)
    if proton_images is not None:
        if augment:
            if return_collapsed:
                proton_images[0], proton_images[collapsed_index] = dual_image_augmenter(proton_images[0], proton_images[collapsed_index], as_channels)
            else:
                proton_images = image_augmenter(proton_images, as_channels)
        if not as_channels:
            if return_collapsed:
                batch_images[0] = batch_images[0].reshape(shape)
                proton_images[0] = proton_images[0].reshape(shape)
            else:
                batch_images = batch_images.reshape(shape)
                proton_images = proton_images.reshape(shape)
        if return_collapsed:
            labels = np.array([True] * (len(batch_images[0])) + [False] * len(proton_images[0]))
            batch_image_label = (np.arange(2) == labels[:, None]).astype(np.float32)
            batch_images[0] = np.concatenate([batch_images[0], proton_images[0]], axis=0)
            if return_features:
                batch_images[feature_index] = np.concatenate([batch_images[feature_index], proton_images[feature_index]], axis=0)
            batch_images[collapsed_index] = np.concatenate([batch_images[collapsed_index], proton_images[collapsed_index]], axis=0)
        else:
            labels = np.array([True] * (len(batch_images)) + [False] * len(proton_images))
            batch_image_label = (np.arange(2) == labels[:, None]).astype(np.float32)
            if return_features:
                batch_images[feature_index] = np.concatenate([batch_images[feature_index], proton_images[feature_index]], axis=0)
            batch_images = np.concatenate([batch_images, proton_images], axis=0)
        if swap:
            if return_features and return_collapsed:
                batch_images[0], batch_images[feature_index], batch_images[collapsed_index], batch_image_label = shuffle(batch_images[0], batch_images[feature_index], batch_images[collapsed_index], batch_image_label)
                batch_image_label = [batch_image_label, batch_image_label, batch_image_label]
            elif return_features and not return_collapsed:
                batch_images[0], batch_images[feature_index], batch_image_label = shuffle(batch_images[0], batch_images[feature_index], batch_image_label)
                batch_image_label = [batch_image_label, batch_image_label]
            elif not return_features and return_collapsed:
                batch_images[0], batch_images[collapsed_index], batch_image_label = shuffle(batch_images[0], batch_images[collapsed_index], batch_image_label)
                batch_image_label = [batch_image_label, batch_image_label]
            else:
                batch_images, batch_image_label = shuffle(batch_images, batch_image_label)
        return batch_images, batch_image_label
    else:
        if positions is not None:
            labels = labels[positions]
        batch_image_label = labels
        if not as_channels:
            if return_collapsed:
                batch_images[0] = batch_images[0].reshape(shape)
            else:
                batch_images = batch_images.reshape(shape)
        if swap:
            if return_features and return_collapsed:
                batch_images[0], batch_images[feature_index], batch_images[collapsed_index], batch_image_label = shuffle(batch_images[0], batch_images[feature_index], batch_images[collapsed_index], batch_image_label)
                batch_image_label = [batch_image_label, batch_image_label, batch_image_label]
            elif return_features and not return_collapsed:
                batch_images[0], batch_images[feature_index], batch_image_label = shuffle(batch_images[0], batch_images[feature_index], batch_image_label)
                batch_image_label = [batch_image_label, batch_image_label]
            elif not return_features and return_collapsed:
                batch_images[0], batch_images[collapsed_index], batch_image_label = shuffle(batch_images[0], batch_images[collapsed_index], batch_image_label)
                batch_image_label = [batch_image_label, batch_image_label]
            else:
                batch_images, batch_image_label = shuffle(batch_images, batch_image_label)
        return batch_images, batch_image_label


def get_random_hdf5_chunk(start, stop, size, time_slice, total_slices, gamma, proton_input=None, labels=None,
                          type_training=None, augment=True, swap=True, shape=None):
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
    :param time_slice: Last index in the time slices
    :param total_slices: Total slices to use starting from time_slice and going earlier
    :return:
    '''

    # Get all possible starting positions given the end point and number of events

    last_possible_start = stop - size

    # Get random starting position
    start_pos = np.random.randint(start, last_possible_start)
    # Range for all positions, to keep with other ones
    positions = range(start_pos, int(start_pos + size))
    with h5py.File(gamma, "r") as images_one:
        if proton_input is not None:
            with h5py.File(proton_input, "r") as images_two:
                proton_data = images_two["Image"]
                training_data = images_one["Image"]
                batch_images = training_data[start_pos:int(start_pos + size), time_slice:time_slice + total_slices, ::]
                proton_images = proton_data[start_pos:int(start_pos + size), time_slice:time_slice + total_slices, ::]
                return common_step(batch_images, positions, labels=labels, proton_images=proton_images, augment=augment,
                                   swap=swap, shape=shape)
        else:
            training_data = images_one["Image"]
            batch_images = training_data[start_pos:int(start_pos + size), time_slice:time_slice + total_slices, ::]
            return common_step(batch_images, positions, labels=labels, augment=augment, swap=swap, shape=shape)


def get_completely_random_hdf5(start, stop, size, time_slice, total_slices, gamma, proton_input=None, labels=None,
                               augment=True, swap=True, shape=None):
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
    positions = sorted(positions)
    with h5py.File(gamma, "r") as images_one:
        if proton_input is not None:
            with h5py.File(proton_input, "r") as images_two:
                proton_data = images_two["Image"]
                training_data = images_one["Image"]
                batch_images = training_data[positions, time_slice:time_slice + total_slices, ::]
                proton_images = proton_data[positions, time_slice:time_slice + total_slices, ::]
                return common_step(batch_images, positions, labels=labels, proton_images=proton_images, augment=augment,
                                   swap=swap, shape=shape)
        else:
            training_data = images_one["Image"]
            batch_images = training_data[positions, time_slice:time_slice + total_slices, ::]
            return common_step(batch_images, positions, labels=labels, augment=augment, swap=swap, shape=shape)


def get_random_from_list(indicies, size, time_slice, total_slices, gamma, proton_input=None, labels=None,
                         augment=True, swap=True, shape=None):
    '''
    Gets a random part of the HDF5 database within a list of given indicies
    This is to help with shuffling data, as currently all the ones come and go in the same
    order
    Does not guarantee that a given event will be used though, unlike before
    Recommended to alternate this with the current one to make sure network has full coverage
    This variant obtains a list of random points, so it will be lower than the other options, but should be better for
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
    positions = sorted(positions)
    with h5py.File(gamma, "r") as images_one:
        if proton_input is not None:
            with h5py.File(proton_input, "r") as images_two:
                proton_data = images_two["Image"]
                training_data = images_one["Image"]
                batch_images = training_data[positions, time_slice:time_slice + total_slices, ::]
                proton_images = proton_data[positions, time_slice:time_slice + total_slices, ::]
                return common_step(batch_images, positions, labels=labels, proton_images=proton_images, augment=augment,
                                   swap=swap, shape=shape)
        else:
            training_data = images_one["Image"]
            batch_images = training_data[positions, time_slice:time_slice + total_slices, ::]
            return common_step(batch_images, positions, labels=labels, augment=augment, swap=swap, shape=shape)


def get_chunk_from_list(indicies, size, time_slice, total_slices, gamma, proton_input=None, labels=None,
                        augment=True, swap=True, shape=None, current_step=0):
    '''
    Gets a section of the HDF5 from the list of indicies, but not randomly,so can iterate through all options
    This is to help with shuffling data, as currently all the ones come and go in the same
    order
    Does not guarantee that a given event will be used though, unlike before
    Recommended to alternate this with the current one to make sure network has full coverage
    This variant obtains a list of random points, so it will be lower than the other options, but should be better for
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
    if (current_step + 1) * size < len(indicies):
        positions = indicies[current_step * size:(current_step + 1) * size]
    else:
        if current_step * size < len(indicies):
            positions = indicies[current_step * size:]
            if len(positions) < size:
                positions += indicies[0:(size - len(positions))]
        else:
            # More overflow
            positions = indicies[0:size]
    positions = sorted(positions)
    with h5py.File(gamma, "r") as images_one:
        if proton_input is not None:
            with h5py.File(proton_input, "r") as images_two:
                proton_data = images_two["Image"]
                training_data = images_one["Image"]
                batch_images = training_data[positions, time_slice:time_slice + total_slices, ::]
                proton_images = proton_data[positions, time_slice:time_slice + total_slices, ::]
                return common_step(batch_images, positions, labels=labels, proton_images=proton_images, augment=augment,
                                   swap=swap, shape=shape)
        else:
            training_data = images_one["Image"]
            batch_images = training_data[positions, time_slice:time_slice + total_slices, ::]
            return common_step(batch_images, positions, labels=labels, augment=augment, swap=swap, shape=shape)


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def true_delta(cog_y, source_y, cog_x, source_x):
    return np.arctan2(
        cog_y - source_y,
        cog_x - source_x
    )


def true_sign(source_x, source_y, cog_x, cog_y, delta):
    true_delta = np.arctan2(
        cog_y - source_y,
        cog_x - source_x,
    )
    true_sign = np.sign(np.abs(delta - true_delta) - np.pi / 2)
    return true_sign


def get_random_from_paths(preprocessor, size, time_slice, total_slices,
                          proton_preprocessor=None, type_training=None, augment=True, swap=True, shape=None,
                          as_channels=False, final_slices=5):
    '''
    Gets a random part of the HDF5 database within start and stop endpoints
    This is to help with shuffling data, as currently all the ones come and go in the same
    order
    Does not guarantee that a given event will be used though, unlike before
    Recommended to alternate this with the current one to make sure network has full coverage
    This variant obtians a list of random points, so it will be lower than the other options, but should be better for
        training

    :param generator:
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

    # For this, the single processors are assumed to infinitely iterate through their files, shuffling the order of the
    # files after every go through of the whole file set, so some kind of shuffling, but not much
    training_data = []
    labels = None
    data_format = {}
    for i in range(size):
        # Call processor size times to get the correct number for the batch
        if as_channels:
            processed_data, data_format = next(preprocessor)
        else:
            processed_data, data_format = next(preprocessor)
        training_data.append(processed_data)
    # Use the type of data to determine what to keep
    if type_training == "Separation":
        training_data = [item[data_format["Image"]] for item in training_data]
    elif type_training == "Energy":
        labels = [item[data_format["Energy"]] for item in training_data]
        labels = np.array(labels)
        training_data = [item[data_format["Image"]] for item in training_data]
    elif type_training == "Disp":
        labels = [euclidean_distance(item[data_format['Source_X']], item[data_format['Source_Y']],
                                     item[data_format['COG_X']], item[data_format['COG_Y']]) for item in training_data]
        labels = np.array(labels)
        training_data = [item[data_format["Image"]] for item in training_data]
    elif type_training == "Sign":
        labels = [true_sign(item[data_format['Source_X']], item[data_format['Source_Y']],
                            item[data_format['COG_X']], item[data_format['COG_Y']], item[data_format['Delta']]) for item
                  in training_data]
        labels = np.array(labels)
        # Create own categorical one since only two sides anyway
        new_labels = np.zeros((labels.shape[0], 2))
        for index, element in enumerate(labels):
            if element < 0:
                new_labels[index][0] = 1.
            else:
                new_labels[index][1] = 1.
        labels = new_labels
        training_data = [item[data_format["Image"]] for item in training_data]

    training_data = np.array(training_data)
    training_data = training_data.reshape(-1, training_data.shape[2], training_data.shape[3], training_data.shape[4])

    if proton_preprocessor is not None:
        proton_data = []
        for i in range(size):
            # Call processor size times to get the correct number for the batch
            if as_channels:
                processed_data, data_format = next(proton_preprocessor)
            else:
                processed_data, data_format = next(proton_preprocessor)
            proton_data.append(processed_data)
        proton_data = [item[data_format["Image"]] for item in proton_data]
        proton_data = np.array(proton_data)
        proton_data = proton_data.reshape(-1, proton_data.shape[2], proton_data.shape[3], proton_data.shape[4])
        if not as_channels:
            batch_images = training_data[::, time_slice:time_slice + total_slices, ::]
            proton_images = proton_data[::, time_slice:time_slice + total_slices, ::]
        else:
            batch_images = training_data
            proton_images = proton_data
        return common_step(batch_images, positions=None, labels=labels, proton_images=proton_images, augment=augment,
                           swap=swap, shape=shape, as_channels=as_channels)
    else:
        if not as_channels:
            batch_images = training_data[::, time_slice:time_slice + total_slices, ::]
        else:
            batch_images = training_data
        return common_step(batch_images, positions=None, labels=labels, augment=augment, swap=swap, shape=shape,
                           as_channels=as_channels)


def augment_image_batch(images, proton_images=None, type_training=None, augment=False, swap=True, shape=None,
                        as_channels=False, return_collapsed=False, return_features=False):
    """
    This is for use with the eventfile_generator, given a set of images, return the possibly augmented ones and labels
    :param images:
    :param proton_images:
    :param type_training:
    :param augment:
    :param swap:
    :param shape:
    :param as_channels:
    :param final_slices:
    :return:
    """

    # For this, the single processors are assumed to infinitely iterate through their files, shuffling the order of the
    # files after every go through of the whole file set, so some kind of shuffling, but not much
    # Get the correct index for the collapsed and feature data if used
    if return_features and return_collapsed:
        feature_index = 2
        collapsed_index = 3
    elif return_features and not return_collapsed:
        feature_index = 2
        collapsed_index = -99
    elif not return_features and return_collapsed:
        feature_index = -99
        collapsed_index = 2
    else:
        feature_index = -99
        collapsed_index = -99

    labels = None
    data_format = images[0][1]
    training_data = [item[0] for item in images]
    if return_features:
        features_list = [item[feature_index] for item in images]
    if return_collapsed:
        collapsed_list = [item[collapsed_index] for item in images]

    # Use the type of data to determine what to keep
    if type_training == "Separation":
        training_data = [item[data_format["Image"]] for item in training_data]
    elif type_training == "Energy":
        labels = [item[data_format["Energy"]] for item in training_data]
        labels = np.array(labels)
    elif type_training == "Disp":
        labels = [euclidean_distance(item[data_format['Source_X']], item[data_format['Source_Y']],
                                     item[data_format['COG_X']], item[data_format['COG_Y']]) for item in training_data]
        labels = np.array(labels)
        training_data = [item[data_format["Image"]] for item in training_data]
    elif type_training == "Sign":
        labels = [true_sign(item[data_format['Source_X']], item[data_format['Source_Y']],
                            item[data_format['COG_X']], item[data_format['COG_Y']], item[data_format['Delta']]) for item
                  in training_data]
        training_data = [item[data_format["Image"]] for item in training_data]
        labels = np.array(labels)
        # Create own categorical one since only two sides anyway
        new_labels = np.zeros((labels.shape[0], 2))
        for index, element in enumerate(labels):
            if element < 0:
                new_labels[index][0] = 1.
            else:
                new_labels[index][1] = 1.
        labels = new_labels
        training_data = [item[data_format["Image"]] for item in training_data]

    training_data = np.array(training_data)
    training_data = training_data.reshape(-1, training_data.shape[2], training_data.shape[3], training_data.shape[4])

    if return_collapsed or return_features:
        batch_images = [training_data]
    else:
        batch_images = training_data

    if return_features:
        features = np.array(features_list)
        batch_images.append(features)

    if return_collapsed:
        # Gather collapsed data
        collapsed_image = np.array(collapsed_list)
        collapsed_image = collapsed_image.reshape(-1, collapsed_image.shape[3], collapsed_image.shape[4], 1)
        batch_images.append(collapsed_image)

    if proton_images is not None:
        proton_data = [item[0] for item in proton_images]
        if return_features:
            proton_features_list = [item[feature_index] for item in proton_images]
        if return_collapsed:
            proton_collapsed_list = [item[collapsed_index] for item in proton_images]
        proton_data = [item[data_format["Image"]] for item in proton_data]
        proton_data = np.array(proton_data)
        proton_data = proton_data.reshape(-1, proton_data.shape[2], proton_data.shape[3], proton_data.shape[4])
        if return_collapsed or return_features:
            proton_images = [proton_data]
        else:
            proton_images = proton_data

        if return_features:
            features = np.array(proton_features_list)
            proton_images.append(features)

        if return_collapsed:
            # Gather collapsed data
            collapsed_image = np.array(proton_collapsed_list)
            collapsed_image = collapsed_image.reshape(-1, collapsed_image.shape[3], collapsed_image.shape[4], 1)
            proton_images.append(collapsed_image)

        return common_step(batch_images, positions=None, labels=labels, proton_images=proton_images, augment=augment,
                           swap=swap, shape=shape, as_channels=as_channels, return_features=return_features, return_collapsed=return_collapsed)
    else:
        return common_step(batch_images, positions=None, labels=labels, augment=augment, swap=swap, shape=shape,
                           as_channels=as_channels, return_collapsed=return_collapsed, return_features=return_features)
