# to force on CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, Activation, Flatten, ConvLSTM2D, Conv2D, MaxPooling2D
import numpy as np
from threedinputs.library.baseStuff import trainModel, testAndPlotModel
import keras.backend as K
import tensorflow as tf

model = keras.models.load_model("/run/media/jacob/WDRed8Tb1/Models/3DEnergy/Drop_0.45LSTM_0.4Patch_3Time_35EndTime_40Strides_2.h5")


testAndPlotModel(model, batch_size=4, type_model="Energy", time_slice=40, total_slices=35,
                 path_mc_images="/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_Gamma_TimInfo_Images.h5",)