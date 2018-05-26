import numpy as np
import h5py

path_mc_images = ""
training_fraction = 0.6
validation_fraction = 0.2
testing_fraction = 0.2

with h5py.File(path_mc_images, 'r') as f2:
    length_items = len(f2['Image'])
    length_training = training_fraction * length_items
    length_validate = (training_fraction + validation_fraction)*length_items


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def testingYielder(path_to_training_data, type_training, percent_training, time_slice=40, total_slices=25,
                 num_events_per_epoch=1000, path_to_proton_data=None, batch_size=64):
    with h5py.File(path_to_training_data, 'r') as f:
        items = list(f.items())[1][1].shape[0]
        items = int(items * percent_training)
        length_dataset = len(f['Image'])
        section = 0
        offset = int(section * num_events_per_epoch)
        times_train_in_items = int(np.floor(items / num_events_per_epoch))
        validation_test = validation_fraction * items
        num_batch_in_validate = int(validation_test / batch_size)
        image = f['Image']
        if type_training == 'Energy':
            energy = f['Energy']
            while True:
                batch_num = 0
                section = section % num_batch_in_validate
                offset = int(section * num_batch_in_validate)
                while batch_size * (batch_num + 1) < items:
                    batch_images = image[int(offset + int((batch_num) * batch_size)):int(
                        offset + int((batch_num + 1) * batch_size))]
                    # Now slice it to only take the first 40 frames of the trigger from Jan's analysis
                    batch_images = batch_images[:, time_slice - total_slices:time_slice, ::]
                    batch_image_label = energy[int(offset + int((batch_num) * batch_size)):int(
                        offset + int((batch_num + 1) * batch_size))]
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
                offset = int(section * num_batch_in_validate)
                while batch_size * (batch_num + 1) < items:
                    batch_images = image[int(offset + int((batch_num) * batch_size)):int(
                        offset + int((batch_num + 1) * batch_size))]
                    # Now slice it to only take the first 40 frames of the trigger from Jan's analysis
                    batch_images = batch_images[:, time_slice - total_slices:time_slice, ::]
                    source_x_tmp = source_x[int(offset + int((batch_num) * batch_size)):int(
                        offset + int((batch_num + 1) * batch_size))]
                    source_y_tmp = source_y[int(offset + int((batch_num) * batch_size)):int(
                        offset + int((batch_num + 1) * batch_size))]
                    cog_x_tmp = cog_x[int(offset + int((batch_num) * batch_size)):int(
                        offset + int((batch_num + 1) * batch_size))]
                    cog_y_tmp = cog_y[int(offset + int((batch_num) * batch_size)):int(
                        offset + int((batch_num + 1) * batch_size))]
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
                            batch_images = image[int(offset + batch_num * batch_size):int(
                                offset + (batch_num + 1) * batch_size)]
                            proton_images = proton_data[int(offset + batch_num * batch_size):int(
                                offset + (batch_num + 1) * batch_size)]
                            labels = np.array([True] * (len(batch_images)) + [False] * len(proton_images))
                            batch_image_label = (np.arange(2) == labels[:, None]).astype(np.float32)
                            batch_images = np.concatenate([batch_images, proton_images], axis=0)

                            batch_num += 1
                            yield (batch_images, batch_image_label)
                        section += 1

def validationYielder(path_to_training_data, type_training, percent_training, time_slice=40, total_slices=25,
                 num_events_per_epoch=1000, path_to_proton_data=None, batch_size=64):
    with h5py.File(path_to_training_data, 'r') as f:
        items = list(f.items())[1][1].shape[0]
        items = int(items * percent_training)
        length_dataset = len(f['Image'])
        section = 0
        offset = int(section * num_events_per_epoch)
        times_train_in_items = int(np.floor(items / num_events_per_epoch))
        validation_test = validation_fraction * items
        num_batch_in_validate = int(validation_test / batch_size)
        image = f['Image']
        if type_training == 'Energy':
            energy = f['Energy']
            while True:
                batch_num = 0
                section = section % num_batch_in_validate
                offset = int(section * num_batch_in_validate)
                while batch_size * (batch_num + 1) < items:
                    batch_images = image[int(length_training + offset + int((batch_num) * batch_size)):int(
                        length_training + offset + int((batch_num + 1) * batch_size))]
                    # Now slice it to only take the first 40 frames of the trigger from Jan's analysis
                    batch_images = batch_images[:, time_slice - total_slices:time_slice, ::]
                    batch_image_label = energy[int(length_training + offset + int((batch_num) * batch_size)):int(
                        length_training + offset + int((batch_num + 1) * batch_size))]
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
                offset = int(section * num_batch_in_validate)
                while batch_size * (batch_num + 1) < items:
                    batch_images = image[int(length_training + offset + int((batch_num) * batch_size)):int(
                        length_training + offset + int((batch_num + 1) * batch_size))]
                    # Now slice it to only take the first 40 frames of the trigger from Jan's analysis
                    batch_images = batch_images[:, time_slice - total_slices:time_slice, ::]
                    source_x_tmp = source_x[int(length_training + offset + int((batch_num) * batch_size)):int(
                        length_training + offset + int((batch_num + 1) * batch_size))]
                    source_y_tmp = source_y[int(length_training + offset + int((batch_num) * batch_size)):int(
                        length_training + offset + int((batch_num + 1) * batch_size))]
                    cog_x_tmp = cog_x[int(length_training + offset + int((batch_num) * batch_size)):int(
                        length_training + offset + int((batch_num + 1) * batch_size))]
                    cog_y_tmp = cog_y[int(length_training + offset + int((batch_num) * batch_size)):int(
                        length_training + offset + int((batch_num + 1) * batch_size))]
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
                            batch_images = image[int(length_training + offset + batch_num * batch_size):int(
                                length_training + offset + (batch_num + 1) * batch_size)]
                            proton_images = proton_data[int(length_training + offset + batch_num * batch_size):int(
                                length_training + offset + (batch_num + 1) * batch_size)]
                            labels = np.array([True] * (len(batch_images)) + [False] * len(proton_images))
                            batch_image_label = (np.arange(2) == labels[:, None]).astype(np.float32)
                            batch_images = np.concatenate([batch_images, proton_images], axis=0)

                            batch_num += 1
                            yield (batch_images, batch_image_label)
                        section += 1

def testingYielder(path_to_training_data, type_training, percent_training, time_slice=40, total_slices=25,
                      num_events_per_epoch=1000, path_to_proton_data=None, batch_size=64):
    with h5py.File(path_to_training_data, 'r') as f:
        items = list(f.items())[1][1].shape[0]
        items = int(items * percent_training)
        length_dataset = len(f['Image'])
        section = 0
        offset = int(section * num_events_per_epoch)
        times_train_in_items = int(np.floor(items / num_events_per_epoch))
        validation_test = validation_fraction * items
        num_batch_in_validate = int(validation_test / batch_size)
        image = f['Image']
        if type_training == 'Energy':
            energy = f['Energy']
            while True:
                batch_num = 0
                section = section % num_batch_in_validate
                offset = int(section * num_batch_in_validate)
                while batch_size * (batch_num + 1) < items:
                    batch_images = image[int(length_validate + offset + int((batch_num) * batch_size)):int(
                        length_validate + offset + int((batch_num + 1) * batch_size))]
                    # Now slice it to only take the first 40 frames of the trigger from Jan's analysis
                    batch_images = batch_images[:, time_slice - total_slices:time_slice, ::]
                    batch_image_label = energy[int(length_validate + offset + int((batch_num) * batch_size)):int(
                        length_validate + offset + int((batch_num + 1) * batch_size))]
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
                offset = int(section * num_batch_in_validate)
                while batch_size * (batch_num + 1) < items:
                    batch_images = image[int(length_validate + offset + int((batch_num) * batch_size)):int(
                        length_validate + offset + int((batch_num + 1) * batch_size))]
                    # Now slice it to only take the first 40 frames of the trigger from Jan's analysis
                    batch_images = batch_images[:, time_slice - total_slices:time_slice, ::]
                    source_x_tmp = source_x[int(length_validate + offset + int((batch_num) * batch_size)):int(
                        length_validate + offset + int((batch_num + 1) * batch_size))]
                    source_y_tmp = source_y[int(length_validate + offset + int((batch_num) * batch_size)):int(
                        length_validate + offset + int((batch_num + 1) * batch_size))]
                    cog_x_tmp = cog_x[int(length_validate + offset + int((batch_num) * batch_size)):int(
                        length_validate + offset + int((batch_num + 1) * batch_size))]
                    cog_y_tmp = cog_y[int(length_validate + offset + int((batch_num) * batch_size)):int(
                        length_validate + offset + int((batch_num + 1) * batch_size))]
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
                            batch_images = image[int(length_validate + offset + batch_num * batch_size):int(
                                length_validate + offset + (batch_num + 1) * batch_size)]
                            proton_images = proton_data[int(length_validate + offset + batch_num * batch_size):int(
                                length_validate + offset + (batch_num + 1) * batch_size)]
                            labels = np.array([True] * (len(batch_images)) + [False] * len(proton_images))
                            batch_image_label = (np.arange(2) == labels[:, None]).astype(np.float32)
                            batch_images = np.concatenate([batch_images, proton_images], axis=0)

                            batch_num += 1
                            yield (batch_images, batch_image_label)
                        section += 1