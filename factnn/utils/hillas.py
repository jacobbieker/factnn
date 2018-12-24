import photon_stream_analysis as phs_analysis
from photon_stream_analysis import reject
import photon_stream as phs
from os.path import basename
from os.path import join
import numpy as np


def extract_simulation_features(phs_file, corsika_file, out_dir):
    """
    Extracts Hillas and other parameters from simulation files
    :param phs_file:
    :param corsika_file:
    :param out_dir:
    :return:
    """
    out_path = join(out_dir, basename(phs_file).split('.')[0]+'.ft.msg')
    triggered, thrown = phs_analysis.extract.from_simulation(
        phs_path=phs_file, mmcs_corsika_path=corsika_file
    )
    phs_analysis.extract.write_simulation_extraction(
        triggered=triggered, thrown=thrown,
        out_path=out_path
    )
    return 1

def extract_single_simulation_features(event):
    """
    Extracts features from a single PHS simulation events and returns them
    :param phs_event: PHS simulation event
    :return:
    """
    cluster = phs.PhotonStreamCluster(event.photon_stream)
    cluster = reject.early_or_late_clusters(cluster)
    features = phs_analysis.extract.raw_features(photon_stream=event.photon_stream, cluster=cluster)
    features['type'] = phs.io.binary.SIMULATION_EVENT_TYPE_KEY
    features['az'] = np.deg2rad(event.az)
    features['zd'] = np.deg2rad(event.zd)

    features['run'] = event.simulation_truth.run
    features['event'] = event.simulation_truth.event
    features['reuse'] = event.simulation_truth.reuse

    features['particle'] = event.simulation_truth.air_shower.particle
    features['energy'] = event.simulation_truth.air_shower.energy
    features['theta'] = event.simulation_truth.air_shower.theta
    features['phi'] = event.simulation_truth.air_shower.phi
    features['impact_x'] = event.simulation_truth.air_shower.impact_x(event.simulation_truth.reuse)
    features['impact_y'] = event.simulation_truth.air_shower.impact_y(event.simulation_truth.reuse)
    features['starting_altitude'] = event.simulation_truth.air_shower.starting_altitude
    features['height_of_first_interaction'] = event.simulation_truth.air_shower.height_of_first_interaction

    return features


def extract_observation_features(phs_file, out_dir):
    """
    Extracts Hillas and other parameters from observation files
    :param phs_file:
    :param out_dir:
    :return:
    """
    out_path = join(out_dir, basename(phs_file).split('.')[0]+'.ft.msg')
    triggered, thrown = phs_analysis.extract.from_observation(
        phs_path=phs_file,
    )
    phs_analysis.extract.write_observation_extraction(
        triggered=triggered,
        out_path=out_path
    )
    return 1

def extract_single_observation_features(event):
    """
    Extracts features from a single PHS observation event and returns them
    :param phs_event: PHS observation event
    :return:
    """
    cluster = phs.PhotonStreamCluster(event.photon_stream)
    cluster = reject.early_or_late_clusters(cluster)
    features = phs_analysis.extract.raw_features(photon_stream=event.photon_stream, cluster=cluster)
    features['type'] = phs.io.binary.OBSERVATION_EVENT_TYPE_KEY
    features['az'] = np.deg2rad(event.az)
    features['zd'] = np.deg2rad(event.zd)

    features['night'] = event.observation_info.night
    features['run'] = event.observation_info.run
    features['event'] = event.observation_info.event

    features['time'] = event.observation_info._time_unix_s + event.observation_info._time_unix_us/1e6

    return features
