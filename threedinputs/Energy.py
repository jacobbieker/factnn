# to force on CPU
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ConvLSTM2D, Conv2D, MaxPooling2D
import numpy as np
from threedinputs.library.baseStuff import trainAndTestModel


def create_model(patch_size, dropout_layer, lstm_dropout, time_slices, strides):
    # Make the model
    model = Sequential()

    # Base Conv layer
    model.add(ConvLSTM2D(32, kernel_size=patch_size, strides=strides,
                         padding='same',
                         input_shape=(time_slices, 75, 75, 1), activation='relu', dropout=dropout_layer, recurrent_dropout=lstm_dropout, recurrent_activation='hard_sigmoid'))
    #model.add(
    #    ConvLSTM2D(32, kernel_size=3, strides=2,
    #               padding='same', activation='relu', dropout=0.3, recurrent_dropout=0.4, recurrent_activation='hard_sigmoid'))

    #model.add(Dropout(dropout_layer))

    model.add(
        Conv2D(64, patch_size, strides=strides,
               padding='same'))
    model.add(Activation('relu'))
    #model.add(MaxPooling2D())
    model.add(Dropout(dropout_layer))
    model.add(
        Conv2D(128, patch_size, strides=strides,
               padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_layer))
    #model.add(MaxPooling2D())
    model.add(Flatten())

    for i in range(1):
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(dropout_layer/2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout_layer/2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(dropout_layer/2))

    # Final Dense layer
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse',
                  metrics=['mae'])

    return model


# Now do MC go through of parameters

for i in range(30):
    dropout_layer = np.round(np.random.uniform(0.0, 1.0), 2)
    lstm_dropout = np.round(np.random.uniform(0.0, 1.0), 2)
    batch_size = 32
    patch_size = np.random.randint(0, 6)
    time_slices = np.random.randint(5,100)
    end_slice = np.random.randint(time_slices, 100)
    strides = np.random.randint(1,4)
    model = create_model(patch_size, dropout_layer, lstm_dropout, time_slices=time_slices, strides=strides)
    model_name = "/run/media/jacob/WDRed8Tb1/Models/3DEnergy/" + "Drop_" + str(dropout_layer) + "LSTM_" + str(lstm_dropout) +\
                 "Patch_" + str(patch_size) + "Time_" + str(time_slices) + "Strides_" + str(strides)
    trainAndTestModel(model, batch_size=batch_size, num_epochs=400, type_model="Energy", time_slice=end_slice, total_slices=time_slices,
                      model_name=model_name, path_mc_images="/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_Gamma_TimInfo_Images.h5",
                      )
