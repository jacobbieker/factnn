# to force on CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ConvLSTM2D, Conv2D, Conv3D, MaxPooling3D, MaxPooling2D, LSTM, Reshape
import keras.backend as K
import tensorflow as tf
import numpy as np
from threedinputs.library.baseStuff import trainModel, testAndPlotModel


def create_model(patch_size, dropout_layer, lstm_dropout, time_slices, strides):
    # Make the model
    # Make the model
    model = Sequential()
    # Base Conv layer
    model.add(ConvLSTM2D(32, kernel_size=patch_size, strides=strides,
                         padding='same',
                         input_shape=(time_slices, 75, 75, 1), activation='relu', dropout=dropout_layer/2, recurrent_dropout=lstm_dropout/2, recurrent_activation='hard_sigmoid', return_sequences=True))
    model.add(ConvLSTM2D(64, kernel_size=patch_size, strides=strides,
                         padding='same', activation='relu', dropout=dropout_layer/2, recurrent_dropout=lstm_dropout/2, recurrent_activation='hard_sigmoid', return_sequences=True))
    #model.add(MaxPooling2D())
    model.add(
        Conv3D(64, kernel_size=patch_size, strides=strides,
               padding='same', activation='relu'))
    model.add(MaxPooling3D())
    model.add(
        Conv3D(64, kernel_size=patch_size, strides=strides,
               padding='same', activation='relu'))
    model.add(MaxPooling3D())
    #model.add(
    #    Conv2D(128, kernel_size=patch_size, strides=strides,
    #           padding='same', activation='relu'))
    #model.add(MaxPooling2D())
    #model.add(Dropout(0.3))
    #model.add(
    #    Conv3D(128, kernel_size=(5,3,3), strides=2,
    #               padding='same', activation='relu'))
    #model.add(Dropout(0.3))
    #model.add(BatchNormalization())

    #model.add(Dropout(dropout_layer))
    model.add(Flatten())

    for i in range(1):
        #model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout_layer/2))
        #model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(dropout_layer/2))

    # Final Dense layer
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['acc'])

    return model


# Now do MC go through of parameters

for i in range(30):
    try:
        dropout_layer = np.round(np.random.uniform(0.0, 0.6), 2)
        lstm_dropout = np.round(np.random.uniform(0.0, 0.6), 2)
        batch_size = 8
        patch_size = np.random.randint(3, 6)
        time_slices = 35#np.random.randint(5,100)
        end_slice = 40#np.random.randint(time_slices, 100)
        strides = np.random.randint(1,3)
        model = create_model(patch_size, dropout_layer, lstm_dropout, time_slices=time_slices, strides=strides)
        model_name = "/run/media/jacob/WDRed8Tb1/Models/3DSep/" + "Drop_" + str(dropout_layer) + "LSTM_" + str(lstm_dropout) + \
                     "Patch_" + str(patch_size) + "Time_" + str(time_slices) + "EndTime_" + str(end_slice) + "Strides_" + str(strides)
        trainModel(model, batch_size=batch_size, num_epochs=400, training_fraction=0.001, validation_fraction=0.001, type_model="Separation", time_slice=end_slice, total_slices=time_slices,
                   model_name=model_name, path_mc_images="/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_Gamma_TimInfo_Images.h5",
                   path_proton_images="/run/media/jacob/WDRed8Tb1/Rebinned_5_MC_Proton_Timing_Images.h5")
        testAndPlotModel(model, batch_size=batch_size, type_model="Separation", time_slice=end_slice, total_slices=time_slices,
                         path_mc_images="/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_Gamma_TimInfo_Images.h5",
                         path_proton_images="/run/media/jacob/WDRed8Tb1/Rebinned_5_MC_Proton_Timing_Images.h5")
    except Exception as e:
        print(e)
        pass
    K.clear_session()
    tf.reset_default_graph()
