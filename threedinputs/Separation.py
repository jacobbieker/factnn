# to force on CPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ConvLSTM2D, Conv2D, MaxPooling2D

from threedinputs.library.baseStuff import trainAndTestModel


# Make the model
model = Sequential()

# Base Conv layer
model.add(ConvLSTM2D(32, kernel_size=3, strides=2,
                     padding='same',
                     input_shape=(35, 75, 75, 1), activation='relu', dropout=0.3, recurrent_dropout=0.4, recurrent_activation='hard_sigmoid'))
#model.add(
#    ConvLSTM2D(32, kernel_size=3, strides=2,
#               padding='same', activation='relu', dropout=0.3, recurrent_dropout=0.4, recurrent_activation='hard_sigmoid'))

#model.add(Dropout(dropout_layer))

model.add(
    Conv2D(64, (3,3), strides=(1, 1),
           padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(
    Conv2D(128, (3,3), strides=(1, 1),
           padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Flatten())

for i in range(1):
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(1/2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(1/2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(1/2))

# Final Dense layer
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse',
              metrics=['mae'])

trainAndTestModel(model, batch_size=32, num_epochs=1600, type_model="Energy", time_slice=40, total_slices=35,
                  model_name="./EnergyTesting", path_mc_images="/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_Gamma_TimInfo_Images.h5",
                  )
