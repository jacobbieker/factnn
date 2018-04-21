import fact_plots
import os
import fact
import yaml
from fact.io import read_h5py
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


def plot_skymap(df, width=4, bins=100, center_ra=None, center_dec=None, ax=None):
    '''
    Plot a 2d histogram of the reconstructed positions of air showers

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame of the reconstructed events containing the columns
        `reconstructed_source_position_0`, `reconstructed_source_position_0`,
        `zd_tracking`, `az_tracking`, `time`, where time is the
        observation time as datetime
    width: float
        Extent of the plot in degrees
    bins: int
        number of bins
    center_ra: float
        right ascension of the center in degrees
    center_dec: float
        declination of the center in degrees
    ax: matplotlib.axes.Axes
        axes to plot into
    '''
    ax = ax or plt.gca()

    ra = df['ra_prediction']
    dec = df['dec_prediction']

    if center_ra is None:
        center_ra = ra.mean()
        center_ra *= 15  # conversion from hourangle to degree

    if center_dec is None:
        center_dec = dec.mean()

    bins, x_edges, y_deges, img = ax.hist2d(
        ra * 15,  # conversion from hourangle to degree
        dec,
        bins=bins,
        range=[
            [center_ra - width / 2, center_ra + width / 2],
            [center_dec - width / 2, center_dec + width / 2]
        ],
    )

    ax.set_xlabel('right ascencion / degree')
    ax.set_ylabel('declination / degree')
    ax.set_aspect(1)

    return ax, img


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

    events = read_h5py(data_path, key='events', columns=columns)

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


main("/run/media/jacob/WDRed8Tb1/dl2_theta/precuts/crab_precuts.hdf5", source='CRAB')

main("/run/media/jacob/WDRed8Tb1/dl2_theta/precuts/Mrk421_precuts.hdf5")

main("/run/media/jacob/WDRed8Tb1/dl2_theta/precuts/Mrk501_precuts.hdf5")

# convert to ra and dec, bin it, and then run unet on it to find possible sources

# Add in theta to get the angles? And off regions in the camera, need that to determine significance
# Could just do a radial one maybe?

# But use ra and dec of a source from SkyCoord as the truth, with an area around it for UNET, convert all pixel things to ra and dec and plaster over each other
#

try:
    # df = read_h5py("/run/media/jacob/WDRed8Tb1/dl2_theta/precuts/crab_precuts.hdf5", key='events')

    print("Read in file")

    # plot_skymap(df=df)
except:
    try:
        df = read_h5py("/run/media/jacob/WDRed8Tb1/dl2_theta/precuts/Mrk421_precuts.hdf5", key='events')

        print("Read in file")

        plot_skymap(df=df)
    except:
        print("This Sucks")
