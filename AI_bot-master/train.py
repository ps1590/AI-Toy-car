import numpy as np
import soundfile as sf

# Library for machine learning
import keras
from keras.utils import to_categorical
from keras.models import Sequential,Model
from keras.layers import MaxPooling2D,Dropout,Flatten,Dense,LSTM,MaxPooling1D
from librosa.feature import mfcc 
import tensorflow as tf

npz = np.load('out/data_cent.npz')
data = npz['data']
label = npz['label']

# print the shapes of the input and label arrays
print(data.shape)
print(label.shape)
########################################################################################

# Def Loss function

# this is the implementation of categorical loss function from scratch
# the input to the function is predicted probability and a one hot vector
# it computes the loss by summing over the expression -ylog(p) over the tuple
# this is summed over a batch and the final loss is given
def categorical_cross_entropy(ytrue, ypred, axis=-1):
    return -1.0*tf.reduce_mean(tf.reduce_sum(ytrue * tf.log(ypred), axis))

#######################################################################################
# LSTM model

# the model contains a lstm layer to exploit the sequential nature of sound files 
# Followed by maxpooling for eliminating the unnecesary information
# then it is flattened and sent to the dense layer with 5 nodes with softmax as activation 
# layer for probability prediction

model = Sequential()

model.add(LSTM(units = 128, return_sequences = True, input_shape = (data.shape[1],39)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Dense(64, activation='sigmoid'))

model.add(Flatten())
model.add(Dense(5, activation='softmax'))
model.summary() 

# compile the keras model
model.compile(loss=categorical_cross_entropy, optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(data, label, validation_split=0, epochs=20, batch_size=25)


# evaluate the keras model
_, accuracy = model.evaluate(data, label)
print('Accuracy: %f' % (accuracy*100))

model.save('model/model.h5')  # creates a HDF5 file 'model.h5'

