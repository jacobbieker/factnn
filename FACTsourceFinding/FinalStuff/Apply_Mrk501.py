import os
# to force on CPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras import backend as K
import h5py
from fact.io import read_h5py, to_h5py
import yaml
import os
import keras
import numpy as np
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Dense, Dropout, Activation, Conv1D, Flatten, LeakyReLU, Reshape, BatchNormalization, Conv2D, MaxPooling2D
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz
from fact.coordinates import equatorial_to_camera, camera_to_equatorial
from astropy.coordinates.angle_utilities import angular_separation

architecture = 'manjaro'

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb1'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/git-thesis/thesis'

# Total fraction to use per epoch of training data, need inverse
frac_per_epoch = 1
num_epochs = 1000*frac_per_epoch

path_mc_images = "/run/media/jacob/SSD/Rebinned_5_MC_Gamma_BothSource_Images.h5"
path_proton_images = "/run/media/jacob/SSD/Rebinned_5_MC_Proton_BothTracking_Images.h5"
np.random.seed(0)
import pandas as pd
#mrk501df = read_h5py("/home/jacob/open_crab_sample_analysis/build/Mrk501_shrunk.hdf5", key="runs", last=448000)
#to_h5py(mrk501df, "/home/jacob/open_crab_sample_analysis/build/Mrk501_shrunk1_runs.hdf5", key="runs")
#exit()
import h5py
with h5py.File("/home/jacob/open_crab_sample_analysis/open_crab_sample_analysis/build/mrk501_2014_std_analysis_v1.0.0.hdf5", "r+") as f:
    print(len(f["events"]))
exit()
#mrk501df = read_h5py("/run/media/jacob/WDRed8Tb1/open_crab_sample_analysis/build/crab_precuts.hdf5", columns=['unix_time_utc', 'night', 'event_num', 'run_id', 'gamma_energy_prediction', 'source_x_prediction', 'source_y_prediction','pointing_position_zd', 'pointing_position_az', 'gamma_prediction'], key="events")
mrk501df = read_h5py("/home/jacob/open_crab_sample_analysis/open_crab_sample_analysis/build/crab_precuts.hdf5", key="events")
print(mrk501df.columns)
mrk501df['timestamp'] = mrk501df['unix_time_utc_0'] * 1e9 + mrk501df['unix_time_utc_1'] * 1e3
print(pd.to_datetime(mrk501df['timestamp']).dt.to_pydatetime())
# Get mean value of source x and y to estimate the
to_h5py(mrk501df, "/run/media/jacob/WDRed8Tb1/open_crab_sample_analysis/build/Mrk501_precut.hdf5", key="events")
exit()

def get_events_near_crab(path_crab_image, len_images):
    crabdf = read_h5py("/run/media/jacob/WDRed8Tb1/dl2_theta/Mrk501_precuts.hdf5", key="events", last=len_images)
    # now go and get the ones with an ra or dec within 0.16 degrees of the source
    # Source pos in sky AltAz
    to_h5py(crabdf, "Mrk501_shrunk.hdf5", key="events")
    exit()

    coord = SkyCoord.from_name("CRAB")
    center_ra = coord.ra.deg / 15
    center_dec = coord.dec.deg

    # Now have it in ra and dec, convert ra of FACT to degree from hourangle
    #crabdf['ra_prediction']
    #center_ra = crabdf['ra_prediction'].mean()
    #center_dec = crabdf['dec_prediction'].mean()

    # Get the events that have the right ra and dec
    # Give to Gamma/Hadron sep, use Mrk 501 because brighter source
    #df_event = crabdf.index[(crabdf['ra_prediction'] < 1.0025*center_ra) & (crabdf['ra_prediction'] > 0.995*center_ra) & (crabdf['dec_prediction'] < 1.0025*center_dec) & (crabdf['dec_prediction'] > 0.995*center_ra)]
    df_event = crabdf.index[(((crabdf['ra_prediction'] - center_ra)**2) <= 0.025) & (((crabdf['dec_prediction'] - center_dec)**2) <= 0.025)]
    off_event = crabdf.index[(((crabdf['ra_prediction'] - center_ra)**2) > 1.0) | (((crabdf['dec_prediction'] - center_dec)**2) > 1.0)]
    print(df_event)
    # Now get the ones close to the actual thing

    return df_event, off_event

on_evt, off_evt = get_events_near_crab("/run/media/jacob/WDRed8Tb2/Rebinned_5_Crab1314_STDDEV_Images.h5", 000)

path_mc_images = "/run/media/jacob/WDRed8Tb2/Rebinned_5_Crab1314_STDDEV_Images.h5"
import matplotlib.pyplot as plt
from sklearn import metrics
def plot_roc(performace_df, model, ax=None, region="On"):

    ax = ax or plt.gca()

    ax.axvline(0, color='lightgray')
    ax.axvline(1, color='lightgray')
    ax.axhline(0, color='lightgray')
    ax.axhline(1, color='lightgray')

    roc_aucs = []

    mean_fpr, mean_tpr, _ = metrics.roc_curve(performace_df, model)

    ax.set_title(region + ' Area Under Curve: {:.4f}'.format(
        metrics.roc_auc_score(performace_df, model)
    ))

    ax.plot(mean_fpr, mean_tpr, label='ROC curve')
    ax.legend()
    ax.set_aspect(1)

    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.figure.tight_layout()

    return ax

with h5py.File(path_mc_images, 'r') as f:
    # Get some truth data for now, just use Crab images
    items = len(f["Image"])
    images = f['Image'][0:-1]
    print(images.shape)
    #images_false = f['Image'][0:-1]
    temp_train = []
    temp_test = []
    tmp_test_label = []
    tmp_train_label = []
    for index in on_evt:
        temp_train.append(images[index])
        tmp_train_label.append([0,1])
        #tmp_test_label.append([1,0])

    for index in off_evt:
        temp_train.append(images[index])
        tmp_train_label.append([1,0])
        #tmp_test_label.append([1,0])

    train_labels = np.asarray(tmp_train_label)
    validation_dataset = np.asarray(temp_train)
    validation_dataset = validation_dataset
    train_labels = train_labels
    #validating_dataset = np.concatenate([images, images_false], axis=0)
    #print(validating_dataset.shape)
    #labels = np.array([True] * (len(images)) + [False] * len(images_false))
    np.random.seed(0)
    y = validation_dataset
    y_label = train_labels
    print(y.shape)
    print(y_label.shape)
    print("Finished getting data")
from sklearn.metrics import roc_auc_score
from keras.models import load_model
sep_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk("/run/media/jacob/SSD/Development/thesis/FACTsourceFinding/FinalStuff/CrabOnes/")
             for file in fileName if '.h5' in file]
y_label = y_label[:,1].reshape((-1,))
for path in sep_paths:
    print(path)
    #try:
        #if path not in best_auc_sep:
    model = load_model(path)
    predictions = model.predict(y, batch_size=64)
    predictions = predictions[:,1].reshape((-1,))
    print(y_label.shape)
    print(predictions.shape)
    print(roc_auc_score(y_label, predictions))
    fig1 = plt.figure()
    ax = fig1.add_subplot(1,1,1)
    plot_roc(y_label, predictions, ax, "")
    #plt.title("On Region")
    plt.show()
    K.clear_session()
    tf.reset_default_graph()
    #except Exception as e:
    #    print(e)
    #    K.clear_session()
    #    tf.reset_default_graph()
    #    pass


