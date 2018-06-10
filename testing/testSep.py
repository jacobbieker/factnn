import os
# to force on CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import keras
from baseModel.modelTrain import testAndPlotModel
import keras.backend as K
import tensorflow as tf


model = keras.models.load_model("/run/media/jacob/WDRed8Tb1/Models/3DSep/Drop_0.3LSTM_0.14Patch_4Time_25EndTime_30Strides_2.h5")


testAndPlotModel(model, batch_size=8, type_model="Separation", time_slice=30, total_slices=25,
                 path_mc_images="/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_Gamma_TimInfo_Images.h5",
                 path_proton_images="/run/media/jacob/WDRed8Tb1/Rebinned_5_MC_Proton_Timing_Images.h5")
