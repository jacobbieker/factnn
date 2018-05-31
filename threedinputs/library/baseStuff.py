import numpy as np
import h5py
import keras
import keras.backend as K
from sklearn.metrics import r2_score, roc_auc_score
from numpy.random import RandomState
from sklearn.utils import shuffle
from .plotting import plot_probabilities, plot_roc, plot_disp_confusion, plot_energy_confusion
import matplotlib.pyplot as plt


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


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


def training_generator(path_to_training_data, type_training, length_training, time_slice=30, total_slices=25,
                       path_to_proton_data=None, batch_size=64):
    with h5py.File(path_to_training_data, 'r') as f:
        items = length_training
        section = 0
        times_train_in_items = int(np.floor(items / batch_size))
        image = f['Image']
        if type_training == 'Energy':
            energy = f['Energy']
            while True:
                batch_num = 0
                section = section % times_train_in_items
                while batch_size * (batch_num + 1) < items:
                    batch_images = image[int(int((batch_num) * batch_size)):int(
                        int((batch_num + 1) * batch_size)), time_slice - total_slices:time_slice, ::]
                    # Now slice it to only take the first 40 frames of the trigger from Jan's analysis
                    batch_image_label = energy[int(int((batch_num) * batch_size)):int(
                        int((batch_num + 1) * batch_size))]

                    batch_num += 1
                    batch_images = image_augmenter(batch_images)
                    batch_images, batch_image_label = shuffle(batch_images, batch_image_label)
                    yield (batch_images, batch_image_label)
                section += 1
        elif type_training == "Disp":
            source_y = f['Source_X']
            source_x = f['Source_Y']
            cog_x = f['COG_X']
            cog_y = f['COG_Y']
            while True:
                batch_num = 0
                section = section % times_train_in_items
                while batch_size * (batch_num + 1) < items:
                    batch_images = image[int(int((batch_num) * batch_size)):int(
                        int((batch_num + 1) * batch_size)), time_slice - total_slices:time_slice, ::]
                    # Now slice it to only take the first 40 frames of the trigger from Jan's analysis
                    source_x_tmp = source_x[int(int((batch_num) * batch_size)):int(
                        int((batch_num + 1) * batch_size))]
                    source_y_tmp = source_y[int(int((batch_num) * batch_size)):int(
                        int((batch_num + 1) * batch_size))]
                    cog_x_tmp = cog_x[int(int((batch_num) * batch_size)):int(
                        int((batch_num + 1) * batch_size))]
                    cog_y_tmp = cog_y[int(int((batch_num) * batch_size)):int(
                        int((batch_num + 1) * batch_size))]
                    batch_image_label = euclidean_distance(
                        source_x_tmp, source_y_tmp,
                        cog_x_tmp, cog_y_tmp
                    )
                    batch_num += 1
                    # Can rotate, etc. the image because not finding the source x,y, but the distance, and that would
                    # be the same if the whole thing was rotated
                    #batch_images = image_augmenter(batch_images)
                    batch_images, batch_image_label = shuffle(batch_images, batch_image_label)
                    yield (batch_images, batch_image_label)
                section += 1

        elif type_training == "Sign":
            source_y = f['Source_X']
            source_x = f['Source_Y']
            cog_x = f['COG_X']
            cog_y = f['COG_Y']
            delta = f['Delta']

        elif type_training == "Separation":
            if path_to_proton_data is None:
                print("Error: No Proton File")
                exit(-1)
            else:
                with h5py.File(path_to_proton_data, 'r') as f2:
                    proton_data = f2['Image']
                    while True:
                        # Now create the batches from labels and other things
                        batch_num = 0
                        section = section % times_train_in_items

                        while batch_size * (batch_num + 1) < items:
                            batch_images = image[int(batch_num * batch_size):int(
                                (batch_num + 1) * batch_size), time_slice - total_slices:time_slice, ::]
                            batch_images = image_augmenter(batch_images)
                            proton_images = proton_data[int(batch_num * batch_size):int(
                                (batch_num + 1) * batch_size), time_slice - total_slices:time_slice, ::]
                            proton_images = image_augmenter(proton_images)
                            labels = np.array([True] * (len(batch_images)) + [False] * len(proton_images))
                            batch_image_label = (np.arange(2) == labels[:, None]).astype(np.float32)
                            batch_images = np.concatenate([batch_images, proton_images], axis=0)

                            batch_num += 1
                            batch_images, batch_image_label = shuffle(batch_images, batch_image_label)
                            yield (batch_images, batch_image_label)
                        section += 1


def validation_generator(path_to_training_data, type_training, length_validation, length_training, time_slice=30,
                         total_slices=25, path_to_proton_data=None, batch_size=64):
    with h5py.File(path_to_training_data, 'r') as f:
        items = length_validation
        section = 0
        times_train_in_items = int(np.floor(items / batch_size))
        num_batch_in_validate = int(length_validation / batch_size)
        image = f['Image']
        if type_training == 'Energy':
            energy = f['Energy']
            while True:
                batch_num = 0
                section = section % num_batch_in_validate
                while batch_size * (batch_num + 1) < items:
                    batch_images = image[int(length_training + int((batch_num) * batch_size)):int(
                        length_training + int((batch_num + 1) * batch_size)), time_slice - total_slices:time_slice, ::]
                    # Now slice it to only take the first 40 frames of the trigger from Jan's analysis
                    batch_image_label = energy[int(length_training + int((batch_num) * batch_size)):int(
                        length_training + int((batch_num + 1) * batch_size))]
                    batch_num += 1
                    batch_images = image_augmenter(batch_images)
                    yield (batch_images, batch_image_label)
                section += 1
        elif type_training == "Disp":
            source_y = f['Source_X']
            source_x = f['Source_Y']
            cog_x = f['COG_X']
            cog_y = f['COG_Y']
            while True:
                batch_num = 0
                section = section % num_batch_in_validate
                while batch_size * (batch_num + 1) < items:
                    batch_images = image[int(length_training + int((batch_num) * batch_size)):int(
                        length_training + int((batch_num + 1) * batch_size)), time_slice - total_slices:time_slice, ::]
                    # Now slice it to only take the first 40 frames of the trigger from Jan's analysis
                    source_x_tmp = source_x[int(length_training + int((batch_num) * batch_size)):int(
                        length_training + int((batch_num + 1) * batch_size))]
                    source_y_tmp = source_y[int(length_training + int((batch_num) * batch_size)):int(
                        length_training + int((batch_num + 1) * batch_size))]
                    cog_x_tmp = cog_x[int(length_training + int((batch_num) * batch_size)):int(
                        length_training + int((batch_num + 1) * batch_size))]
                    cog_y_tmp = cog_y[int(length_training + int((batch_num) * batch_size)):int(
                        length_training + int((batch_num + 1) * batch_size))]
                    batch_image_label = euclidean_distance(
                        source_x_tmp, source_y_tmp,
                        cog_x_tmp, cog_y_tmp
                    )
                    batch_num += 1
                    #batch_images = image_augmenter(batch_images)
                    yield (batch_images, batch_image_label)
                section += 1

        elif type_training == "Sign":
            source_y = f['Source_X']
            source_x = f['Source_Y']
            cog_x = f['COG_X']
            cog_y = f['COG_Y']
            delta = f['Delta']

        elif type_training == "Separation":
            if path_to_proton_data is None:
                print("Error: No Proton File")
                exit(-1)
            else:
                with h5py.File(path_to_proton_data, 'r') as f2:
                    proton_data = f2['Image']
                    while True:
                        # Now create the batches from labels and other things
                        batch_num = 0
                        section = section % times_train_in_items

                        while batch_size * (batch_num + 1) < items:
                            batch_images = image[int(length_training + batch_num * batch_size):int(
                                length_training + (batch_num + 1) * batch_size), time_slice - total_slices:time_slice,
                                           ::]
                            proton_images = proton_data[int(length_training + batch_num * batch_size):int(
                                length_training + (batch_num + 1) * batch_size), time_slice - total_slices:time_slice,
                                            ::]
                            labels = np.array([True] * (len(batch_images)) + [False] * len(proton_images))
                            batch_image_label = (np.arange(2) == labels[:, None]).astype(np.float32)
                            batch_images = np.concatenate([batch_images, proton_images], axis=0)

                            batch_num += 1
                            batch_images = image_augmenter(batch_images)
                            yield (batch_images, batch_image_label)
                        section += 1


def testing_generator(path_to_training_data, type_training, length_validate, length_testing, time_slice=30,
                      total_slices=25, path_to_proton_data=None, batch_size=64):
    with h5py.File(path_to_training_data, 'r') as f:
        items = length_testing
        section = 0
        times_train_in_items = int(np.floor(items / batch_size))
        num_batch_in_validate = int(length_testing / batch_size)
        image = f['Image']
        if type_training == 'Energy':
            energy = f['Energy']
            while True:
                batch_num = 0
                section = section % num_batch_in_validate
                while batch_size * (batch_num + 1) < items:
                    batch_images = image[int(length_validate + int((batch_num) * batch_size)):int(
                        length_validate + int((batch_num + 1) * batch_size)), time_slice - total_slices:time_slice, ::]
                    # Now slice it to only take the first 40 frames of the trigger from Jan's analysis
                    batch_image_label = energy[int(length_validate + int((batch_num) * batch_size)):int(
                        length_validate + int((batch_num + 1) * batch_size))]
                    batch_num += 1
                    yield (batch_images, batch_image_label)
                section += 1
        elif type_training == "Disp":
            source_y = f['Source_X']
            source_x = f['Source_Y']
            cog_x = f['COG_X']
            cog_y = f['COG_Y']
            while True:
                batch_num = 0
                section = section % num_batch_in_validate
                while batch_size * (batch_num + 1) < items:
                    batch_images = image[int(length_validate + int((batch_num) * batch_size)):int(
                        length_validate + int((batch_num + 1) * batch_size)), time_slice - total_slices:time_slice, ::]
                    # Now slice it to only take the first 40 frames of the trigger from Jan's analysis
                    source_x_tmp = source_x[int(length_validate + int((batch_num) * batch_size)):int(
                        length_validate + int((batch_num + 1) * batch_size))]
                    source_y_tmp = source_y[int(length_validate + int((batch_num) * batch_size)):int(
                        length_validate + int((batch_num + 1) * batch_size))]
                    cog_x_tmp = cog_x[int(length_validate + int((batch_num) * batch_size)):int(
                        length_validate + int((batch_num + 1) * batch_size))]
                    cog_y_tmp = cog_y[int(length_validate + int((batch_num) * batch_size)):int(
                        length_validate + int((batch_num + 1) * batch_size))]
                    batch_image_label = euclidean_distance(
                        source_x_tmp, source_y_tmp,
                        cog_x_tmp, cog_y_tmp
                    )
                    batch_num += 1
                    yield (batch_images, batch_image_label)
                section += 1

        elif type_training == "Sign":
            source_y = f['Source_X']
            source_x = f['Source_Y']
            cog_x = f['COG_X']
            cog_y = f['COG_Y']
            delta = f['Delta']

        elif type_training == "Separation":
            if path_to_proton_data is None:
                print("Error: No Proton File")
                exit(-1)
            else:
                with h5py.File(path_to_proton_data, 'r') as f2:
                    proton_data = f2['Image']
                    while True:
                        # Now create the batches from labels and other things
                        batch_num = 0
                        section = section % times_train_in_items

                        while batch_size * (batch_num + 1) < items:
                            batch_images = image[int(length_validate + batch_num * batch_size):int(
                                length_validate + (batch_num + 1) * batch_size), time_slice - total_slices:time_slice,
                                           ::]
                            proton_images = proton_data[int(length_validate + batch_num * batch_size):int(
                                length_validate + (batch_num + 1) * batch_size), time_slice - total_slices:time_slice,
                                            ::]
                            labels = np.array([True] * (len(batch_images)) + [False] * len(proton_images))
                            batch_image_label = (np.arange(2) == labels[:, None]).astype(np.float32)
                            batch_images = np.concatenate([batch_images, proton_images], axis=0)

                            batch_num += 1
                            yield (batch_images, batch_image_label)
                        section += 1


def trainModel(model, batch_size, num_epochs, type_model,
               time_slice, total_slices, model_name, path_mc_images, path_proton_images=None,
               training_fraction=0.6, validation_fraction=0.2, tensorboard=False):
    if path_proton_images is None:
        with h5py.File(path_mc_images, 'r') as f2:
            length_items = len(f2['Image'])
            length_training = training_fraction * length_items
            length_validate = (training_fraction + validation_fraction) * length_items
            only_length_validate = validation_fraction * length_items
    else:
        with h5py.File(path_proton_images, 'r') as f2:
            length_items = len(f2['Image'])
            length_training = training_fraction * length_items
            length_validate = (training_fraction + validation_fraction) * length_items
            only_length_validate = validation_fraction * length_items

    model_checkpoint = keras.callbacks.ModelCheckpoint(model_name + ".h5",
                                                       monitor='val_loss',
                                                       verbose=0,
                                                       save_best_only=True,
                                                       save_weights_only=False,
                                                       mode='auto', period=1)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                               patience=10,
                                               verbose=0, mode='auto')
    model.summary()
    # Makes it only use
    if tensorboard:
        tb = keras.callbacks.TensorBoard(log_dir='./', histogram_freq=1, batch_size=32, write_graph=True,
                                         write_grads=True,
                                         write_images=False,
                                         embeddings_freq=0,
                                         embeddings_layer_names=None,
                                         embeddings_metadata=None)
        if path_proton_images is not None:
            model.fit_generator(generator=training_generator(path_to_training_data=path_mc_images,
                                                             total_slices=total_slices,
                                                             time_slice=time_slice,
                                                             type_training=type_model,
                                                             batch_size=batch_size,
                                                             path_to_proton_data=path_proton_images,
                                                             length_training=length_training),
                                steps_per_epoch=int(np.floor(training_fraction * length_items / batch_size)),
                                epochs=num_epochs,
                                verbose=1,
                                validation_data=validation_generator(path_to_training_data=path_mc_images,
                                                                     time_slice=time_slice,
                                                                     total_slices=total_slices,
                                                                     batch_size=batch_size,
                                                                     path_to_proton_data=path_proton_images,
                                                                     type_training=type_model,
                                                                     length_training=length_training,
                                                                     length_validation=only_length_validate),
                                callbacks=[early_stop, model_checkpoint, tb],
                                validation_steps=int(np.floor(only_length_validate / batch_size))
                                )
        else:
            model.fit_generator(
                generator=training_generator(path_to_training_data=path_mc_images, total_slices=total_slices,
                                             time_slice=time_slice, type_training=type_model,
                                             batch_size=batch_size, length_training=length_training),
                steps_per_epoch=int(np.floor(training_fraction * length_items / batch_size))
                , epochs=num_epochs,
                verbose=1,
                validation_data=validation_generator(path_to_training_data=path_mc_images, time_slice=time_slice,
                                                     total_slices=total_slices,
                                                     batch_size=batch_size, type_training=type_model,
                                                     length_training=length_training,
                                                     length_validation=only_length_validate),
                callbacks=[early_stop, model_checkpoint, tb],
                validation_steps=int(np.floor(only_length_validate / batch_size))
            )
    else:
        if path_proton_images is not None:
            model.fit_generator(
                generator=training_generator(path_to_training_data=path_mc_images, total_slices=total_slices,
                                             time_slice=time_slice, type_training=type_model,
                                             batch_size=batch_size, path_to_proton_data=path_proton_images,
                                             length_training=length_training),
                steps_per_epoch=int(np.floor(training_fraction * length_items / batch_size))
                , epochs=num_epochs,
                verbose=1,
                validation_data=validation_generator(path_to_training_data=path_mc_images, time_slice=time_slice,
                                                     total_slices=total_slices,
                                                     batch_size=batch_size, path_to_proton_data=path_proton_images,
                                                     type_training=type_model,
                                                     length_training=length_training,
                                                     length_validation=only_length_validate),
                callbacks=[early_stop, model_checkpoint],
                validation_steps=int(np.floor(only_length_validate / batch_size))
            )
        else:
            model.fit_generator(
                generator=training_generator(path_to_training_data=path_mc_images, total_slices=total_slices,
                                             time_slice=time_slice, type_training=type_model,
                                             batch_size=batch_size, length_training=length_training),
                steps_per_epoch=int(np.floor(training_fraction * length_items / batch_size))
                , epochs=num_epochs,
                verbose=1,
                validation_data=validation_generator(path_to_training_data=path_mc_images, time_slice=time_slice,
                                                     total_slices=total_slices,
                                                     batch_size=batch_size, type_training=type_model,
                                                     length_training=length_training,
                                                     length_validation=only_length_validate),
                callbacks=[early_stop, model_checkpoint],
                validation_steps=int(np.floor(only_length_validate / batch_size))
            )
    return model_name + ".h5"


def testAndPlotModel(model, batch_size, time_slice, total_slices, type_model, path_mc_images,
                     path_proton_images=None,
                     training_fraction=0.6, validation_fraction=0.2, testing_fraction=0.2):
    """
    Given a model, and the same inputs as trainModel, test the model and create relevant plots
    :param model: Keras model to test
    :param batch_size:
    :param num_epochs:
    :param type_model:
    :param path_mc_images:
    :param path_proton_images:
    :param training_fraction:
    :param validation_fraction:
    :param testing_fraction:
    :return:
    """
    if path_proton_images is None:
        with h5py.File(path_mc_images, 'r') as f2:
            length_items = len(f2['Image'])
            length_training = training_fraction * length_items
            length_validate = (training_fraction + validation_fraction) * length_items
            testing_length = testing_fraction * length_items
    else:
        with h5py.File(path_proton_images, 'r') as f2:
            length_items = len(f2['Image'])
            length_training = training_fraction * length_items
            length_validate = (training_fraction + validation_fraction) * length_items
            testing_length = testing_fraction * length_items

    if path_proton_images is None:
        # Get the labels by predicting on batches
        generator = testing_generator(path_to_training_data=path_mc_images, time_slice=time_slice,
                                      total_slices=total_slices,
                                      batch_size=batch_size, path_to_proton_data=path_proton_images,
                                      type_training=type_model,
                                      length_validate=length_validate,
                                      length_testing=testing_length)
        steps = int(np.floor(testing_length / batch_size))

        truth = []
        predictions = []
        for i in range(steps):
            # Get each batch and test it
            test_images, test_labels = next(generator)
            test_predictions = model.predict_on_batch(test_images)
            predictions.append(test_predictions)
            truth.append(test_labels)

        predictions = np.asarray(predictions).reshape(-1, )
        truth = np.asarray(truth).reshape(-1, )

        # Now all the labels and predictions made, plot them based on the model type
        if type_model == "Separation":
            fig1 = plt.figure()
            ax = fig1.add_subplot(1, 1, 1)
            plot_roc(truth, predictions, ax=ax)
            fig1.show()
        elif type_model == "Energy":
            score = r2_score(truth, predictions)
            fig1 = plt.figure()
            ax = fig1.add_subplot(1, 1, 1)
            ax.set_title("R^2: {:0.4f}".format(score) + ' Reconstructed vs. True Energy (log color scale)')
            plot_energy_confusion(predictions, truth, ax=ax)
            fig1.show()
        elif type_model == "Disp":
            score = r2_score(truth, predictions)
            fig1 = plt.figure()
            ax = fig1.add_subplot(1, 1, 1)
            ax.set_title("R^2: {:0.4f}".format(score) + ' Reconstructed vs. True Disp')
            plot_disp_confusion(predictions, truth, ax=ax, log_z=False, log_xy=False)
            fig1.show()
