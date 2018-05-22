import numpy as np
import h5py

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def batchYielder(path_to_training_data, type_training, percent_training, num_events_per_epoch=1000, path_to_proton_data=None, batch_size=64):
    with h5py.File(path_to_training_data, 'r') as f:
        items = list(f.items())[1][1].shape[0]
        items = int(items*percent_training)
        length_dataset = len(f['Image'])
        section = 0
        offset = int(section * num_events_per_epoch)
        times_train_in_items = int(np.floor(items / num_events_per_epoch))
        image = f['Image']
        if type_training == 'Energy':
            energy = f['Energy']
        elif type_training == "Disp":
            source_y = f['Source_X']
            source_x = f['Source_Y']
            cog_x = f['COG_X']
            cog_y = f['COG_Y']

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
                batch_images = image[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                if type_training == 'Energy':
                    batch_image_label = energy[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                elif type_training == "Disp":
                    source_x_tmp = source_x[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                    source_y_tmp = source_y[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                    cog_x_tmp = cog_x[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                    cog_y_tmp = cog_y[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                    batch_image_label = euclidean_distance(
                        source_x_tmp, source_y_tmp,
                        cog_x_tmp, cog_y_tmp
                    )
                elif type_training == "Sign":
                    source_x_tmp = source_x[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                    source_y_tmp = source_y[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                    cog_x_tmp = cog_x[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                    cog_y_tmp = cog_y[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                    delta_tmp = delta[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                    true_delta = np.arctan2(
                        source_x_tmp, source_y_tmp,
                        cog_x_tmp, cog_y_tmp
                        )
                    true_sign = np.sign(np.abs(delta_tmp - true_delta) - np.pi / 2)
                    temp_sign = []
                    for sign in true_sign:
                        if sign < 0:
                            temp_sign.append([1,0])
                        else:
                            temp_sign.append([0,1])
                    batch_image_label = np.asarray(temp_sign)
                elif type_training == "Separation":
                    proton_images = proton_data[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                    labels = np.array([True] * (len(batch_images)) + [False] * len(proton_images))
                    batch_image_label = (np.arange(2) == labels[:, None]).astype(np.float32)
                    batch_images = np.concatenate([batch_images, proton_images], axis=0)

                batch_num += 1
                yield (batch_images, batch_image_label)
            section += 1
