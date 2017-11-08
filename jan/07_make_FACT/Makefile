path_raw_crab_folder = /net/big-tank/POOL/projects/fact/photon-stream/pass4/phs/
path_preprocessed_crab_images = /fhgfs/users/jbehnken/make_Data/Crab1314_preprocessed_images.h5
path_crab_runs_to_use = Crab1314_runs_to_use.csv

path_raw_mc_folder = /net/big-tank/POOL/projects/fact/simulation/photon_stream/fact_tools/v.0.18.0/
path_preprocessed_mc_images = /fhgfs/users/jbehnken/make_Data/MC_preprocessed_images.h5

path_raw_mc_diffuse_images = /fhgfs/groups/app/fact/simulated/photon_stream/mc/gammas/diffuse/gamma_gustav_12/
path_preprocessed_mc_diffuse_images = /fhgfs/users/jbehnken/make_Data/MC_diffuse_flat_preprocessed_images.h5
path_preprocessed_mc_diffuse_flat_images = /fhgfs/users/jbehnken/make_Data/MC_diffuse_flat_preprocessed_images.h5

path_Camera_Image_FACT = Camera_Image_FACT.csv
path_hexagonal_mapping_dict = hexagonal_to_quadratic_mapping_dict.p
path_unskewed_camera_plot = build/Unskewed_Camera_Image_FACT_with_hexagonal_pixel.pdf
path_skewed_camera_plot = build/Skewed_Camera_Image_FACT_with_quadratic_pixel.pdf

path_theta = /net/big-tank/POOL/projects/fact/datasets/Crab1314_darknight_std_analysis_0.17.2.hdf5
path_preprocessed_theta = /fhgfs/users/jbehnken/make_Data/Theta_preprocessed.h5

path_architecture_comparison_flag = Architecture_comparison.flag
path_architectures_folder = /fhgfs/users/jbehnken/make_Data/architectures

path_plot_architecture_heatmap = build/Top10_Auc_Architecture_Heatmap.pdf


all:$(path_preprocessed_theta) $(path_preprocessed_crab_images) $(path_preprocessed_mc_images) $(path_preprocessed_mc_diffuse_images) $(path_architecture_comparison_flag) $(path_plot_architecture_heatmap)
	touch Makefile

$(path_preprocessed_theta):$(path_theta) python_files/Preprocess_Crab_Theta.py
	python python_files/Preprocess_Crab_Theta.py $(path_theta) $(path_theta_preprocesses)

$(path_preprocessed_crab_images):$(path_raw_crab_folder) $(path_crab_runs_to_use) $(path_hexagonal_mapping_dict) python_files/Preprocess_Crab_Images.py
	python python_files/Preprocess_Crab_Images.py $(path_raw_crab_folder) $(path_crab_runs_to_use) $(path_hexagonal_mapping_dict) $(path_preprocessed_crab_images)

$(path_hexagonal_mapping_dict):$(path_Camera_Image_FACT) python_files/Create_hexagonal_to_quadratic_mapping_dict.py python_files/Plot_hexagonal_to_quadratic_pixel.py
	python python_files/Create_hexagonal_to_quadratic_mapping_dict.py $(path_Camera_Image_FACT) $(path_hexagonal_mapping_dict)
	python python_files/Plot_hexagonal_to_quadratic_pixel.py $(path_Camera_Image_FACT) $(path_hexagonal_mapping_dict) $(path_unskewed_camera_plot) $(path_skewed_camera_plot)

$(path_preprocessed_mc_images):$(path_raw_mc_folder) $(path_hexagonal_mapping_dict) python_files/Preprocess_MC_Images.py
	python python_files/Preprocess_MC_Images.py $(path_raw_mc_folder) $(path_hexagonal_mapping_dict) $(path_preprocessed_mc_images)
    
$(path_preprocessed_mc_diffuse_images):$(path_raw_mc_folder) $(path_raw_mc_diffuse_images) $(path_hexagonal_mapping_dict) python_files/Preprocess_MC_diffuse_Images.py
	python python_files/Preprocess_MC_diffuse_Images.py $(path_raw_mc_folder) $(path_raw_mc_diffuse_images) $(path_hexagonal_mapping_dict) $(path_preprocessed_mc_diffuse_images)
    
$(path_architecture_comparison_flag):$(path_preprocessed_mc_diffuse_images) python_files/Universal_Hyperparameter_Architectures.py
	python python_files/Universal_Hyperparameter_Architectures.py $(path_preprocessed_mc_diffuse_images) $(path_architectures_folder)
	touch $(path_architecture_comparison_flag)
    
    
$(path_plot_architecture_heatmap):$(path_architecture_comparison_flag) python_files/Plot_architecture_comparison.py
	source activate keras
	python python_files/Plot_architecture_comparison.py $(path_architectures_folder)
	deactivate keras
