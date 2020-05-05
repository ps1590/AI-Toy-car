import numpy as np
import soundfile as sf
import os
import rechelp as rec

# Library for machine learning
import keras
from keras.utils import to_categorical
from keras.models import Sequential,Model,load_model
from keras.layers import MaxPooling2D,Dropout,Flatten,Dense,LSTM,MaxPooling1D
from librosa.feature import mfcc 
import tensorflow as tf
########################################################################################

# Def Loss function

# this is the implementation of categorical loss function from scratch
# the input to the function is predicted probability and a one hot vector
# it computes the loss by summing over the expression -ylog(p) over the tuple
# this is summed over a batch and the final loss is given
def categorical_cross_entropy(ytrue, ypred, axis=-1):
    return -1.0*tf.reduce_mean(tf.reduce_sum(ytrue * tf.log(ypred), axis))
########################################################################################
model = load_model('model/model.h5', custom_objects={'categorical_cross_entropy':categorical_cross_entropy})
os.system('cls' if os.name == 'nt' else 'clear')
print('Model loaded successfully')
#rec.usePyAudio = False
#rec.init(12345)
#rec.btinit(8081)
#rec.btwrite(b'V')

words = ['back', 'forward', 'left', 'right', 'stop']
aw = ['B', 'F', 'L', 'R', 'S']
while True:
	input()
	f = open('testfile.wav', 'wb')
	f.write(rec.record(2))
	f.close()
	new_data = np.zeros([25000,])
	data, samplerate = sf.read('testfile.wav')
	x= len(data)
	p = 25000-x
	y = (25000//2)-(p//2)
	
	for i in range(0,x):
		new_data[y+i] = data[i]
		
	data = (mfcc(y = new_data, sr = samplerate, n_mfcc=39).T)
	data = data.reshape((1,data.shape[0],data.shape[1]))
	#print(data.shape)

	pred = model.predict(data)[0]
	print(pred)
	predInd = np.argmax(pred)
	#rec.btwrite(aw[predInd].encode())
	print(predInd, words[predInd])# if pred[predInd] > 0.998 else print('Not me')
	#rec.btwrite(aw[predInd].encode())
	#while 1:
	'''if predInd == 0:
		rec.btwrite(b'B')
	elif predInd == 1:
		rec.btwrite(b'F')
	elif predInd == 2:
		rec.btwrite(b'L')
	elif predInd == 3:
		rec.btwrite(b'R')
	elif predInd == 4:
		rec.btwrite(b'S')'''
	
