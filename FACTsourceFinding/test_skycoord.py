#import keras
import numpy as np
import h5py
import fact
from fact.io import read_h5py
from fact.coordinates.utils import arrays_to_equatorial, equatorial_to_camera, camera_to_equatorial
import pickle
from astropy.coordinates import SkyCoord
import pandas as pd

from fact.instrument import get_pixel_dataframe

from fact_plots.skymap import plot_skymap
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from astropy.coordinates import SkyCoord

from fact.io import read_h5py

from fact_plots.plotting import add_preliminary
from fact_plots.time import read_timestamp

plot_config = {
    'xlabel': r'$(\theta \,\, / \,\, {}^\circ )^2$',
    'preliminary_position': 'lower center',
    'preliminary_size': 'xx-large',
    'preliminary_color': 'lightgray',
}

columns = [
    'ra_prediction',
    'dec_prediction'
]

thing = SkyCoord.from_name("CRAB")
print(thing)
thing2 = SkyCoord.from_name("MRK421")
print(thing2)
thing3 = SkyCoord.from_name("MRK501")
print(thing3)


def main(data_path, threshold=0.0, key='events', bins=100, width=8.0, preliminary=True, config=None, output=None,
         source=None):
    '''
    Plot a 2d histogram of the origin of the air showers in the
    given hdf5 file in ra, dec.
    '''
    if config:
        with open(config) as f:
            plot_config.update(yaml.safe_load(f))

    if threshold > 0.0:
        columns.append('gamma_prediction')

    events = data_path[columns]

    if threshold > 0.0:
        events = events.query('gamma_prediction >= @threshold').copy()

    fig, ax = plt.subplots(1, 1)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    if source:
        coord = SkyCoord.from_name(source)
        center_ra = coord.ra.deg
        center_dec = coord.dec.deg
    else:
        center_ra = center_dec = None

    ax, img = plot_skymap(
        events,
        width=width,
        bins=bins,
        center_ra=center_ra,
        center_dec=center_dec,
        ax=ax,
    )

    if source:
        ax.plot(
            center_ra,
            center_dec,
            label=source,
            color='r',
            marker='o',
            linestyle='',
            markersize=10,
            markerfacecolor='none',
        )
        ax.legend()

    fig.colorbar(img, cax=cax, label='Gamma-Like Events')

    if preliminary:
        add_preliminary(
            plot_config['preliminary_position'],
            size=plot_config['preliminary_size'],
            color=plot_config['preliminary_color'],
            ax=ax,
            zorder=3,
        )

    fig.tight_layout(pad=0)
    if output:
        fig.savefig(output, dpi=300)
    else:
        plt.show()



def get_obstime(event, night):
    '''
    Return the time from the night and event
    :param event:
    :param night:
    :return:
    '''

    info_df = read_h5py("/run/media/jacob/SSD/Development/thesis/FACTsourceFinding/RunInfo.hdf5", key='info')
    current_event = info_df.loc[(info_df['fNight'] == night) & (info_df['fRunID'] == event)]
    #print(current_event)
    time = current_event['fRunStart'].values
    #print(time)
    return time

def convert_to_hexagonal(image_array):
    '''
    Converts a given image back to hexagonal coordinates, for correct use with ra and dec stuff
    :param image_array:
    :return: List of the photon values for each CHID
    '''

    camera_df = get_pixel_dataframe()
    path_store_mapping_dict = "/run/media/jacob/SSD/Development/thesis/jan/07_make_FACT/quadratic_to_hexagonal_mapping_dict.p"

    # Load the quadratic to hexagonal, to convert back to the correct format again
    id_position = pickle.load(open(path_store_mapping_dict, "rb"))

    chid_values = np.zeros(1440)

    for x in range(image_array.shape[0]):
        for y in range(image_array.shape[1]):
            for i in range(len(id_position)):
                if id_position[i][0] == x and id_position[i][1] == y:
                    chid_values[i] = image_array[x][y]


    return np.array(chid_values)

with h5py.File("/run/media/jacob/WDRed8Tb1/dl2_theta/precuts/Mrk501_precuts.hdf5") as f:
    ra = []
    dec = []
    num_of_each = []
    camera_df = get_pixel_dataframe()
    #items = list(f.items())[0][1].shape[0]
    source_one_images = []
    source_pos_one = []
    tmp_arr = np.zeros(1440)
    #tmp_arr += convert_to_hexagonal(f['Image'][0])
    #source_arr = convert_to_hexagonal(f['Source_Position'][0])

    #chid_source = []
    #for i in range(source_arr.shape[0]):
    #    if source_arr[i] == 1:
    #        chid_source.append(i)
    for index in range(10000):
        print(index)
        # Go through and get every event
        time = get_obstime(f['events']['run_id'][index], f['events']['night'][index])
        source_x = f['events']['source_x_prediction'][index]
        source_y = f['events']['source_y_prediction'][index]
        time = get_obstime(f['events']['run_id'][index], f['events']['night'][index])
        #print(chid_source)
        ra_and_dec = camera_to_equatorial(source_x, source_y, f['events']['zd_source_calc'][index], f['events']['az_source_calc'][index], time)
        #ra_and_dec = camera_to_equatorial(camera_df['x'].values, camera_df['y'].values, f['Zd_deg'][index], f['Az_deg'][index], time)
        # Now have the ra and dec of every pixel, add that to the overall ra and dec predictions one
        ra.append(ra_and_dec[0])
        dec.append(ra_and_dec[1])
        num_of_each.append([1])

    # Now have the ra, dec, and number of photons for each in lists of lists, have to add to that
    import itertools
    ra = np.array(list(itertools.chain.from_iterable(ra)))
    dec = np.array(list(itertools.chain.from_iterable(dec)))
    num_of_each = np.array(list(itertools.chain.from_iterable(num_of_each)))
    all_ra = []
    all_dec = []
    print("All Done, now appending to all")
    for index, element in enumerate(num_of_each):
        count = 0
        while count < element:
            all_ra.append(ra[index])
            all_dec.append(dec[index])
            count += 1

    stuff = {'ra_prediction': all_ra,
             'dec_prediction': all_dec}

    new_df = pd.DataFrame.from_dict(stuff)
    #camera_df['ra_prediction'] = all_ra
    #camera_df['dec_prediction'] = all_dec
    print("Plotting")
    main(new_df, source="MRK501", output="mrk501_skymap_source_calc_prediction_10000.pdf")

    #print(thing3[0][0] / new_stuff[0][0])

    #time = get_obstime(f['Run'][0], f['Night'][0])

#for chid in chid_source:
#     new_stuff = camera_to_equatorial(camera_df['x'][chid], camera_df['y'][chid], f['Zd_deg'][0], f['Az_deg'][0], time)
#     print(new_stuff)


with h5py.File("/run/media/jacob/WDRed8Tb1/FACTSources/Crab_preprocessed_images.h5") as f:
    camera_df = get_pixel_dataframe()
    #items = list(f.items())[0][1].shape[0]
    source_one_images = []
    source_pos_one = []
    tmp_arr = np.zeros(1440)
    ra = []
    dec = []
    num_of_each = []

    #tmp_arr += convert_to_hexagonal(f['Image'][0])
    #source_arr = convert_to_hexagonal(f['Source_Position'][0])

    #chid_source = []
    #for i in range(source_arr.shape[0]):
    #    if source_arr[i] == 1:
    #        chid_source.append(i)
    chid_values = []
    for index in range(1):
        print(index)
        # Go through and get every event
        chid_values = convert_to_hexagonal(f['Image'][index])
        time = get_obstime(f['Run'][index], f['Night'][index])
        camera_df['Photon_Count'] = chid_values
        ra_and_dec = camera_to_equatorial(camera_df['x'].values, camera_df['y'].values, f['Zd_deg'][index], f['Az_deg'][index], time)
        # Now have the ra and dec of every pixel, add that to the overall ra and dec predictions one
        ra.append(ra_and_dec[0])
        dec.append(ra_and_dec[1])
        num_of_each.append(chid_values)

    # Now have the ra, dec, and number of photons for each in lists of lists, have to add to that
    import itertools
    ra = np.array(list(itertools.chain.from_iterable(ra)))
    dec = np.array(list(itertools.chain.from_iterable(dec)))
    num_of_each = np.array(list(itertools.chain.from_iterable(num_of_each)))
    all_ra = []
    all_dec = []
    print("All Done, now appending to all")
    for index, element in enumerate(num_of_each):
        count = 0
        while count < element:
            all_ra.append(ra[index])
            all_dec.append(dec[index])
            count += 1

    stuff = {'ra_prediction': all_ra,
             'dec_prediction': all_dec}

    new_df = pd.DataFrame.from_dict(stuff)
    #camera_df['ra_prediction'] = all_ra
    #camera_df['dec_prediction'] = all_dec
    print("Plotting")
    main(new_df, source="CRAB", output="crab_skymap.pdf")


    #source_x = f['events']['source_position']
    #source_y = f['events']['source_position']
    #time = get_obstime(f['events']['run_id'][10], f['events']['night'][10])
    #print(chid_source)
    #new_stuff = camera_to_equatorial(source_x, source_y, f['events']['zd_pointing'][10], f['events']['az_pointing'][10], time)
    #print(new_stuff)
    #print("Mrk501 Ratio:")
    #print(new_stuff[0][0])


with h5py.File("/run/media/jacob/WDRed8Tb1/dl2_theta/precuts/Mrk501_precuts.hdf5") as f:
    camera_df = get_pixel_dataframe()
    #items = list(f.items())[0][1].shape[0]
    source_one_images = []
    source_pos_one = []
    tmp_arr = np.zeros(1440)
    #tmp_arr += convert_to_hexagonal(f['Image'][0])
    #source_arr = convert_to_hexagonal(f['Source_Position'][0])

    #chid_source = []
    #for i in range(source_arr.shape[0]):
    #    if source_arr[i] == 1:
    #        chid_source.append(i)
    for index in range(100):
        print(index)
        # Go through and get every event
        time = get_obstime(f['Run'][index], f['Night'][index])
        source_x = f['events']['source_position'][index][0]
        source_y = f['events']['source_position'][index][1]
        time = get_obstime(f['events']['run_id'][index], f['events']['night'][index])
        #print(chid_source)
        ra_and_dec = camera_to_equatorial(source_x, source_y, f['events']['zd_pointing'][index], f['events']['az_pointing'][index], time)
        #ra_and_dec = camera_to_equatorial(camera_df['x'].values, camera_df['y'].values, f['Zd_deg'][index], f['Az_deg'][index], time)
        # Now have the ra and dec of every pixel, add that to the overall ra and dec predictions one
        ra.append(ra_and_dec[0])
        dec.append(ra_and_dec[1])
        num_of_each.append(1)

    # Now have the ra, dec, and number of photons for each in lists of lists, have to add to that
    import itertools
    ra = np.array(list(itertools.chain.from_iterable(ra)))
    dec = np.array(list(itertools.chain.from_iterable(dec)))
    num_of_each = np.array(list(itertools.chain.from_iterable(num_of_each)))
    all_ra = []
    all_dec = []
    print("All Done, now appending to all")
    for index, element in enumerate(num_of_each):
        count = 0
        while count < element:
            all_ra.append(ra[index])
            all_dec.append(dec[index])
            count += 1

    stuff = {'ra_prediction': all_ra,
             'dec_prediction': all_dec}

    new_df = pd.DataFrame.from_dict(stuff)
    #camera_df['ra_prediction'] = all_ra
    #camera_df['dec_prediction'] = all_dec
    print("Plotting")
    main(new_df, source="MRK501")#, output="mrk501_skymap.pdf")

    #print(thing3[0][0] / new_stuff[0][0])

    #time = get_obstime(f['Run'][0], f['Night'][0])

   #for chid in chid_source:
   #     new_stuff = camera_to_equatorial(camera_df['x'][chid], camera_df['y'][chid], f['Zd_deg'][0], f['Az_deg'][0], time)
   #     print(new_stuff)

with h5py.File("/run/media/jacob/WDRed8Tb1/dl2_theta/precuts/crab_precuts.hdf5") as f:
    camera_df = get_pixel_dataframe()
    #items = list(f.items())[0][1].shape[0]
    source_one_images = []
    source_pos_one = []
    tmp_arr = np.zeros(1440)
    #tmp_arr += convert_to_hexagonal(f['Image'][0])
    #source_arr = convert_to_hexagonal(f['Source_Position'][0])

    #chid_source = []
    #for i in range(source_arr.shape[0]):
    #    if source_arr[i] == 1:
    #        chid_source.append(i)
    source_x = f['events']['source_position'][10][0]
    source_y = f['events']['source_position'][10][1]
    time = get_obstime(f['events']['run_id'][10], f['events']['night'][10])
    #print(chid_source)
    new_stuff = camera_to_equatorial(source_x, source_y, f['events']['zd_pointing'][10], f['events']['az_pointing'][10], time)
    print(new_stuff)
    print("Crab Ratio:")
    #print(thing[0][0] / new_stuff[0][0])

with h5py.File("/run/media/jacob/WDRed8Tb1/dl2_theta/precuts/Mrk421_precuts.hdf5") as f:
    camera_df = get_pixel_dataframe()
    #items = list(f.items())[0][1].shape[0]
    source_one_images = []
    source_pos_one = []
    tmp_arr = np.zeros(1440)
    #tmp_arr += convert_to_hexagonal(f['Image'][0])
    #source_arr = convert_to_hexagonal(f['Source_Position'][0])

    #chid_source = []
    #for i in range(source_arr.shape[0]):
    #    if source_arr[i] == 1:
    #        chid_source.append(i)
    source_x = f['events']['source_position'][10][0]
    source_y = f['events']['source_position'][10][1]
    time = get_obstime(f['events']['run_id'][10], f['events']['night'][10])
    #print(chid_source)
    new_stuff = camera_to_equatorial(source_x, source_y, f['events']['zd_pointing'][10], f['events']['az_pointing'][10], time)
    print(new_stuff)
    print("Mrk421 Ratio:")
    #print(thing2[0][0] / new_stuff[0][0])