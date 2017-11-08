import pandas as pd
import numpy as np
import h5py
import os



prediction_path = '/fhgfs/users/jbehnken/make_Data/crab1314_prediction.csv'
plotting_path = '/fhgfs/users/jbehnken/make_Data/Theta_Plotting.h5'
image_path = '/home/jbehnken/07_make_FACT/build/Theta_Plot.pdf'


def theta_square_plot(theta2_cut = 0.8, data_path = plotting_path, key = 'events', start = None, end = None, threshold = 0.5, bins = 40, alpha = 0.2, output = False):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import h5py
    from dateutil.parser import parse as parse_date

    from fact.io import read_h5py
    from fact.analysis import (li_ma_significance, split_on_off_source_dependent,)
    import click

    columns = [
        'gamma_prediction',
        'theta_deg',
        'theta_deg_off_1',
        'theta_deg_off_2',
        'theta_deg_off_3',
        'theta_deg_off_4',
        'theta_deg_off_5',
        'unix_time_utc',
    ]

    stats_box_template = r'''Source: {source}, $t_\mathrm{{obs}} = {t_obs:.2f}\,\mathrm{{h}}$
    $N_\mathrm{{On}} = {n_on}$, $N_\mathrm{{Off}} = {n_off}$, $\alpha = {alpha}$
    $N_\mathrm{{Exc}} = {n_excess:.1f} \pm {n_excess_err:.1f}$, $S_\mathrm{{Li&Ma}} = {significance:.1f}\,\sigma$
    '''


    theta_cut = np.sqrt(theta2_cut)

    with h5py.File(data_path, 'r') as f:
        source_dependent = 'gamma_prediction_off_1' in f[key].keys()

    if source_dependent:
        print('Separation was using source dependent features')
        columns.extend('gamma_prediction_off_' + str(i) for i in range(1, 6))
        theta_cut = np.inf
        theta2_cut = np.inf

    events = read_h5py(data_path, key='events', columns=columns)
    events['timestamp'] = pd.to_datetime(events['unix_time_utc_0'] * 1e6 + events['unix_time_utc_1'],unit='us',)
    runs = read_h5py(data_path, key='runs')
    runs['run_start'] = pd.to_datetime(runs['run_start'])
    runs['run_stop'] = pd.to_datetime(runs['run_stop'])

    if start is not None:
        events = events.query('timestamp >= @start')
        runs = runs.query('run_start >= @start')
    if end is not None:
        events = events.query('timestamp <= @end')
        runs = runs.query('run_stop <= @end')

    if source_dependent:
        on_data, off_data = split_on_off_source_dependent(events, threshold)
        theta_on = on_data.theta_deg
        theta_off = off_data.theta_deg
    else:
        selected = events.query('gamma_prediction >= {}'.format(threshold))
        theta_on = selected.theta_deg
        theta_off = pd.concat([selected['theta_deg_off_{}'.format(i)]for i in range(1, 6)])

    del events

    if source_dependent:
        limits = [0,max(np.percentile(theta_on, 99)**2, np.percentile(theta_off, 99)**2),]
    else:
        limits = [0, 0.3]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    h_on, bin_edges = np.histogram(theta_on.apply(lambda x: x**2).values,bins=bins,range=limits)
    h_off, bin_edges, _ = ax.hist(theta_off.apply(lambda x: x**2).values,bins=bin_edges,range=limits,weights=np.full(len(theta_off), 0.2),histtype='stepfilled',color='lightgray',)

    bin_center = bin_edges[1:] - np.diff(bin_edges) * 0.5
    bin_width = np.diff(bin_edges)

    ax.errorbar(bin_center,h_on,yerr=np.sqrt(h_on) / 2,xerr=bin_width / 2,linestyle='',label='On',)
    ax.errorbar(bin_center,h_off,yerr=alpha * np.sqrt(h_off) / 2,xerr=bin_width / 2,linestyle='',label='Off',)

    if not source_dependent:
        ax.axvline(theta_cut**2, color='gray', linestyle='--')

    n_on = np.sum(theta_on < theta_cut)
    n_off = np.sum(theta_off < theta_cut)
    significance = li_ma_significance(n_on, n_off, alpha=alpha)


    ax.text(0.5, 0.95,stats_box_template.format(source='Crab',t_obs=83.656,n_on=n_on, n_off=n_off, alpha=alpha,n_excess=n_on - alpha * n_off,n_excess_err=np.sqrt(n_on + alpha**2 * n_off),significance=significance,),transform=ax.transAxes,fontsize=12,va='top',ha='center',)

    ax.set_xlabel(r'$(\theta / {}^\circ )^2$')
    ax.legend()
    fig.tight_layout()
    plt.xlim(0.0, 0.3)

    if output:
        fig.savefig(output, dpi=300)
    else:
        #plt.show()
        pass


theta_square_plot(theta2_cut=0.026, data_path=plotting_path, key='events', start=None, end=None, threshold=0.766, bins=60, alpha=0.2, output=image_path)
