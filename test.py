import h5py

def print_attrs(name, obj):
    print(name)
    #for key, val in obj.attrs.iteritems():
    #    print("    %s: %s" % (key, val))

#f = h5py.File('../../open_crab_sample_facttools_dl2.hdf5','r')
f = h5py.File('../../gamma_simulations_facttools_dl2.hdf5','r')
#f = h5py.File('predictions.hdf','r')

f.visititems(print_attrs)