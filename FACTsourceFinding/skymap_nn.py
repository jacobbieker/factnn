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

def main(data_path, threshold=0.0, key='events', bins=100, width=4.0, preliminary=True, config=None, output=None, source=None):
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


main("/run/media/jacob/WDRed8Tb1/dl2_theta/precuts/crab_precuts.hdf5")

main("/run/media/jacob/WDRed8Tb1/dl2_theta/precuts/Mrk421_precuts.hdf5")

main("/run/media/jacob/WDRed8Tb1/dl2_theta/precuts/Mrk501_precuts.hdf5")

try:
    #df = read_h5py("/run/media/jacob/WDRed8Tb1/dl2_theta/precuts/crab_precuts.hdf5", key='events')

    print("Read in file")

    #plot_skymap(df=df)
except:
    try:
        df = read_h5py("/run/media/jacob/WDRed8Tb1/dl2_theta/precuts/Mrk421_precuts.hdf5", key='events')

        print("Read in file")

        plot_skymap(df=df)
    except:
        print("This Sucks")