# to force on CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import keras
from factnn.data.modelTrain import testAndPlotModel
import keras.backend as K
import tensorflow as tf

print("Two Model System")

model = keras.models.load_model("/run/media/jacob/WDRed8Tb1/Models/3DEnergy/Drop_0.01LSTM_0.8Patch_4Time_35EndTime_40Strides_4.h5")


testAndPlotModel(model, batch_size=4, type_model="Energy", time_slice=40, total_slices=35,
                 path_mc_images="/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_Gamma_TimInfo_Images.h5",)
exit()
K.clear_session()
tf.reset_default_graph()


model = keras.models.load_model("/run/media/jacob/WDRed8Tb1/Models/3DEnergy/Drop_0.64LSTM_0.05Patch_3Time_35EndTime_40Strides_4.h5")


testAndPlotModel(model, batch_size=4, type_model="Energy", time_slice=40, total_slices=35,
                 path_mc_images="/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_Gamma_TimInfo_Images.h5",)

K.clear_session()
tf.reset_default_graph()
exit()

model = keras.models.load_model("/run/media/jacob/WDRed8Tb1/Models/3DEnergy/Drop_0.5LSTM_0.3Patch_3Time_35EndTime_40Strides_2.h5")


testAndPlotModel(model, batch_size=4, type_model="Energy", time_slice=40, total_slices=35,
                 path_mc_images="/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_Gamma_TimInfo_Images.h5",)

K.clear_session()
tf.reset_default_graph()

model = keras.models.load_model("/run/media/jacob/WDRed8Tb1/Models/3DEnergy/Drop_0.67LSTM_0.59Patch_2Time_25EndTime_30Strides_2.h5")


testAndPlotModel(model, batch_size=4, type_model="Energy", time_slice=30, total_slices=25,
                 path_mc_images="/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_Gamma_TimInfo_Images.h5",)

K.clear_session()
tf.reset_default_graph()


model = keras.models.load_model("/run/media/jacob/WDRed8Tb1/Models/3DEnergy/Drop_0.27LSTM_0.74Patch_1Time_25EndTime_40Strides_3.h5")


testAndPlotModel(model, batch_size=4, type_model="Energy", time_slice=40, total_slices=25,
                 path_mc_images="/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_Gamma_TimInfo_Images.h5",)