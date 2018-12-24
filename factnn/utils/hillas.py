import photon_stream_analysis as phs_analysis
import photon_stream as phs
from os.path import basename
from os.path import join


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