import numpy as np
import h5py
import keras
import keras.backend as K
from sklearn.metrics import r2_score, roc_auc_score


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def trainingYielder(path_to_training_data, type_training, length_training, time_slice=30, total_slices=25,
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
                         int((batch_num + 1) * batch_size))]
                    # Now slice it to only take the first 40 frames of the trigger from Jan's analysis
                    batch_images = batch_images[:, time_slice - total_slices:time_slice, ::]
                    batch_image_label = energy[int( int((batch_num) * batch_size)):int(
                         int((batch_num + 1) * batch_size))]
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
                section = section % times_train_in_items
                while batch_size * (batch_num + 1) < items:
                    batch_images = image[int(int((batch_num) * batch_size)):int(
                         int((batch_num + 1) * batch_size))]
                    # Now slice it to only take the first 40 frames of the trigger from Jan's analysis
                    batch_images = batch_images[:, time_slice - total_slices:time_slice, ::]
                    source_x_tmp = source_x[int( int((batch_num) * batch_size)):int(
                        int((batch_num + 1) * batch_size))]
                    source_y_tmp = source_y[int( int((batch_num) * batch_size)):int(
                         int((batch_num + 1) * batch_size))]
                    cog_x_tmp = cog_x[int( int((batch_num) * batch_size)):int(
                         int((batch_num + 1) * batch_size))]
                    cog_y_tmp = cog_y[int( int((batch_num) * batch_size)):int(
                         int((batch_num + 1) * batch_size))]
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
                            batch_images = image[int( batch_num * batch_size):int(
                                 (batch_num + 1) * batch_size)]
                            proton_images = proton_data[int( batch_num * batch_size):int(
                                 (batch_num + 1) * batch_size)]
                            labels = np.array([True] * (len(batch_images)) + [False] * len(proton_images))
                            batch_image_label = (np.arange(2) == labels[:, None]).astype(np.float32)
                            batch_images = np.concatenate([batch_images, proton_images], axis=0)

                            batch_num += 1
                            yield (batch_images, batch_image_label)
                        section += 1


def validationYielder(path_to_training_data, type_training, length_validation, length_training, time_slice=30,
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
                        length_training  + int((batch_num + 1) * batch_size))]
                    # Now slice it to only take the first 40 frames of the trigger from Jan's analysis
                    batch_images = batch_images[:, time_slice - total_slices:time_slice, ::]
                    batch_image_label = energy[int(length_training + int((batch_num) * batch_size)):int(
                        length_training + int((batch_num + 1) * batch_size))]
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
                    batch_images = image[int(length_training + int((batch_num) * batch_size)):int(
                        length_training  + int((batch_num + 1) * batch_size))]
                    # Now slice it to only take the first 40 frames of the trigger from Jan's analysis
                    batch_images = batch_images[:, time_slice - total_slices:time_slice, ::]
                    source_x_tmp = source_x[int(length_training + int((batch_num) * batch_size)):int(
                        length_training + int((batch_num + 1) * batch_size))]
                    source_y_tmp = source_y[int(length_training + int((batch_num) * batch_size)):int(
                        length_training  + int((batch_num + 1) * batch_size))]
                    cog_x_tmp = cog_x[int(length_training + int((batch_num) * batch_size)):int(
                        length_training + int((batch_num + 1) * batch_size))]
                    cog_y_tmp = cog_y[int(length_training + int((batch_num) * batch_size)):int(
                        length_training + int((batch_num + 1) * batch_size))]
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
                            batch_images = image[int(length_training + batch_num * batch_size):int(
                                length_training + (batch_num + 1) * batch_size)]
                            proton_images = proton_data[int(length_training + batch_num * batch_size):int(
                                length_training + (batch_num + 1) * batch_size)]
                            labels = np.array([True] * (len(batch_images)) + [False] * len(proton_images))
                            batch_image_label = (np.arange(2) == labels[:, None]).astype(np.float32)
                            batch_images = np.concatenate([batch_images, proton_images], axis=0)

                            batch_num += 1
                            yield (batch_images, batch_image_label)
                        section += 1


def testingYielder(path_to_training_data, type_training, length_validate, length_testing, time_slice=30,
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
                        length_validate + int((batch_num + 1) * batch_size))]
                    # Now slice it to only take the first 40 frames of the trigger from Jan's analysis
                    batch_images = batch_images[:, time_slice - total_slices:time_slice, ::]
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
                        length_validate + int((batch_num + 1) * batch_size))]
                    # Now slice it to only take the first 40 frames of the trigger from Jan's analysis
                    batch_images = batch_images[:, time_slice - total_slices:time_slice, ::]
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
                                length_validate + (batch_num + 1) * batch_size)]
                            proton_images = proton_data[int(length_validate + batch_num * batch_size):int(
                                length_validate + (batch_num + 1) * batch_size)]
                            labels = np.array([True] * (len(batch_images)) + [False] * len(proton_images))
                            batch_image_label = (np.arange(2) == labels[:, None]).astype(np.float32)
                            batch_images = np.concatenate([batch_images, proton_images], axis=0)

                            batch_num += 1
                            yield (batch_images, batch_image_label)
                        section += 1


def trainAndTestModel(model, batch_size, num_epochs, type_model,
                      time_slice, total_slices, model_name, path_mc_images, path_proton_images=None,
                      training_fraction=0.6, validation_fraction=0.2, tensorboard=False):
    with h5py.File(path_mc_images, 'r') as f2:
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
    if tensorboard:
        tb = keras.callbacks.TensorBoard(log_dir='./', histogram_freq=1, batch_size=32, write_graph=True,
                                         write_grads=True,
                                         write_images=False,
                                         embeddings_freq=0,
                                         embeddings_layer_names=None,
                                         embeddings_metadata=None)

    model.summary()
    # Makes it only use
    if tensorboard:
        if path_proton_images is not None:
            model.fit_generator(generator=trainingYielder(path_to_training_data=path_mc_images, total_slices=total_slices,
                                                          time_slice=time_slice, type_training=type_model,
                                                          batch_size=batch_size, path_to_proton_data=path_proton_images, length_training=length_training),
                                steps_per_epoch=int(np.floor(training_fraction * length_items / batch_size))
                                , epochs=num_epochs,
                                verbose=1, validation_data=validationYielder(validation_fraction, time_slice=time_slice,
                                                                             total_slices=total_slices,
                                                                             batch_size=batch_size, path_to_proton_data=path_proton_images, type_training=type_model,
                                                                             length_training=length_training,
                                                                             length_validation=only_length_validate),
                                callbacks=[early_stop, model_checkpoint, tb],
                                )
        else:
            model.fit_generator(generator=trainingYielder(path_to_training_data=path_mc_images, total_slices=total_slices,
                                                          time_slice=time_slice, path_to_proton_data=path_proton_images, type_training=type_model,
                                                          batch_size=batch_size, length_training=length_training),
                                steps_per_epoch=int(np.floor(training_fraction * length_items / batch_size))
                                , epochs=num_epochs,
                                verbose=1, validation_data=validationYielder(validation_fraction, time_slice=time_slice,
                                                                             total_slices=total_slices, path_to_proton_data=path_proton_images,
                                                                             batch_size=batch_size, type_training=type_model,
                                                                             length_training=length_training,
                                                                             length_validation=only_length_validate),
                                callbacks=[early_stop, model_checkpoint, tb],
                                )
    else:
        if path_proton_images is not None:
            model.fit_generator(generator=trainingYielder(path_to_training_data=path_mc_images, total_slices=total_slices,
                                                          time_slice=time_slice, type_training=type_model,
                                                          batch_size=batch_size, path_to_proton_data=path_proton_images, length_training=length_training),
                                steps_per_epoch=int(np.floor(training_fraction * length_items / batch_size))
                                , epochs=num_epochs,
                                verbose=1, validation_data=validationYielder(validation_fraction, time_slice=time_slice,
                                                                             total_slices=total_slices,
                                                                             batch_size=batch_size, path_to_proton_data=path_proton_images, type_training=type_model,
                                                                             length_training=length_training,
                                                                             length_validation=only_length_validate),
                                callbacks=[early_stop, model_checkpoint],
                                )
        else:
            model.fit_generator(generator=trainingYielder(path_to_training_data=path_mc_images, total_slices=total_slices,
                                                          time_slice=time_slice, path_to_proton_data=path_proton_images, type_training=type_model,
                                                          batch_size=batch_size, length_training=length_training),
                                steps_per_epoch=int(np.floor(training_fraction * length_items / batch_size))
                                , epochs=num_epochs,
                                verbose=1, validation_data=validationYielder(validation_fraction, time_slice=time_slice,
                                                                             total_slices=total_slices, path_to_proton_data=path_proton_images,
                                                                             batch_size=batch_size, type_training=type_model,
                                                                             length_training=length_training,
                                                                             length_validation=only_length_validate),
                                callbacks=[early_stop, model_checkpoint],
                                )
    return model_name + ".h5"
