# to force on CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.models import load_model

model = load_model("/home/jacob/FACT-NN-Analysis-0.1/misc/timeModels/bestModels/Drop_0.3LSTM_0.14Patch_4Time_25EndTime_30Strides_2.h5") # Classification
#model = load_model("/home/jacob/PublicFACTModels/_MC_Disp3DSpatial_p_(5, 5)_drop_0.21_numDense_0_conv_2_pool_0_denseN_211_convN_124.h5") # Disp
#model = load_model("/home/jacob/PublicFACTModels/Drop_0.45LSTM_0.4Patch_3Time_35EndTime_40Strides_2.h5") # Either Disp or Energy
#model = load_model("/home/jacob/PublicFACTModels/Drop_0.27LSTM_0.74Patch_1Time_25EndTime_40Strides_3.h5") # Either Disp or Energy
model.summary()

for layer in model.layers:
    print(layer.input_shape)