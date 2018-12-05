from factnn import ProtonPreprocessor, GammaPreprocessor, GammaDiffusePreprocessor
import os
from multiprocessing import Pool
from functools import partial


def convert_to_eventfiles(directory, output_path, file_type, clump_size=20, clean_images=True, dl2_file=None, threads=4):
    # Get paths from the directories
    source_paths = []
    for source_dir in directory:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith("phs.jsonl.gz"):
                    source_paths.append(os.path.join(root, file))

    def f(clump_size, path):
        print("Gamma")
        print("Size: ", clump_size)
        gamma_configuration = {
            'paths': [path],
            'dl2_file': dl2_file
        }

        if file_type == "Gamma":
            preprocessor = GammaPreprocessor(config=gamma_configuration)
            preprocessor.event_processor(os.path.join(output_path, "gamma"), clean_images=clean_images, clump_size=clump_size)
        elif file_type == "Proton":
            preprocessor = ProtonPreprocessor(config=gamma_configuration)
            preprocessor.event_processor(os.path.join(output_path, "proton"), clean_images=clean_images, clump_size=clump_size)
        elif file_type == "Diffuse":
            preprocessor = GammaDiffusePreprocessor(config=gamma_configuration)
            preprocessor.event_processor(os.path.join(output_path, "gamma_diffuse"), clean_images=clean_images)

    output_paths = [os.path.join(output_path, "proton", "no_clean"),os.path.join(output_path, "proton", "clump"+str(clump_size)),os.path.join(output_path, "proton", "core"+str(clump_size)),
                    os.path.join(output_path, "gamma", "no_clean"),os.path.join(output_path, "gamma", "clump"+str(clump_size)),os.path.join(output_path, "gamma", "core"+str(clump_size))]
    for path in output_paths:
        if not os.path.exists(path):
            os.makedirs(path)

    pool = Pool(threads)
    func = partial(f, clump_size)
    p = pool.map_async(func, source_paths)

    return p


def get_paths(directory):
    return NotImplementedError

