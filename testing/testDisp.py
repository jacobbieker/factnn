# to force on CPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import keras
from io import testAndPlotModel
import keras.backend as K
import tensorflow as tf


model = keras.models.load_model("/run/media/jacob/WDRed8Tb1/Models/3DDisp/_MC_Disp3DSpatial_p_(5, 5)_drop_0.21_numDense_0_conv_2_pool_0_denseN_211_convN_124.h5")
#model = keras.models.load_model("/run/media/jacob/WDRed8Tb1/Models/3DDisp/_MC_Disp3DSpatial_p_(5, 5)_drop_0.6_numDense_4_conv_4_pool_0_denseN_106_convN_89.h5")
model.summary()
#exit()
testAndPlotModel(model, batch_size=8, type_model="Disp", time_slice=40, total_slices=25,
                 path_mc_images="/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_Gamma_Diffuse_TimInfo_Images.h5",)

K.clear_session()
tf.reset_default_graph()

model = keras.models.load_model("/run/media/jacob/WDRed8Tb1/Models/3DDisp/_MC_Disp3DSpatial_p_(5, 5)_drop_0.6_numDense_4_conv_4_pool_0_denseN_106_convN_89.h5")
model.summary()
#exit()
testAndPlotModel(model, batch_size=8, type_model="Disp", time_slice=40, total_slices=25,
                 path_mc_images="/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_Gamma_Diffuse_TimInfo_Images.h5",)

K.clear_session()
tf.reset_default_graph()

model = keras.models.load_model("/run/media/jacob/WDRed8Tb1/Models/3DDisp/_MC_Disp3DSpatial_p_(5, 5)_drop_0.6_numDense_4_conv_4_pool_0_denseN_106_convN_89.h5")
model.summary()
#exit()
testAndPlotModel(model, batch_size=8, type_model="Disp", time_slice=40, total_slices=25, validation_fraction=0.0, testing_fraction=0.4,
                 path_mc_images="/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_Gamma_Diffuse_TimInfo_Images.h5",)

K.clear_session()
tf.reset_default_graph()

model = keras.models.load_model("/run/media/jacob/WDRed8Tb1/Models/3DDisp/_MC_Disp3DSpatial_p_(5, 5)_drop_0.21_numDense_0_conv_2_pool_0_denseN_211_convN_124.h5")
model.summary()
#exit()
testAndPlotModel(model, batch_size=8, type_model="Disp", time_slice=40, total_slices=25, validation_fraction=0.0, testing_fraction=0.4,
                 path_mc_images="/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_Gamma_Diffuse_TimInfo_Images.h5",)