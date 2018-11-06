import h5py


def create_dataset(preprocessor, output_file="output.hdf5"):
    """
    Create an HDF5 dataset from a preprocessor's output
    :param preprocessor: The preprocessor to use, outputs data, data_format on each call of next
    :param output_file: Name of the output file
    :return:
    """
    gen = preprocessor.batch_processor()
    batch, data_format = next(gen)
    formatted_batch = preprocessor.format(batch)
    row_count = formatted_batch[data_format["Image"]].shape[0]
    print(row_count)

    with h5py.File(output_file, 'w') as hdf:
        # Now go through each key in data_format, create the dataset
        dset_sets = [None for i in formatted_batch]
        for key, value in data_format:
            maxshape = (None,) + formatted_batch[value].shape[1:]
            dset = hdf.create_dataset(key,
                                      shape=formatted_batch[key].shape,
                                      maxshape=maxshape,
                                      chunks=formatted_batch[key].shape,
                                      dtype=formatted_batch[key].dtype)
            dset_sets[value] = dset
            dset_sets[value][:] = formatted_batch[key]

        for batch in gen:
            batch, data_format = batch
            formatted_batch = preprocessor.format(batch)

            shape = formatted_batch[1].shape[0]
            for index, dset in enumerate(dset_sets):
                dset.resize(row_count + shape, axis=0)
                dset[row_count:] = formatted_batch[index]

            row_count += shape
