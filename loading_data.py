from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.utils import to_categorical
import random


def load_mnist(n_init = 10):
	"""
	load mnist in a particuliar way, with 4 different partition  Train, Test, Valid, Pool
	Train will have n_init image of each class
	Test 5000
	Valid 5000
	Pool the rest

	Return X_train,Y_train,X_test,Y_test,X_valid,Y_valid,X_pool,Y_pool
	"""
	 # the data, shuffled and split between train and test sets
	(X_train_All, y_train_All), (X_test, y_test) = mnist.load_data()

	X_train_All = X_train_All[:,:,:,None]
	X_test = X_test[:,:,:,None]
	
	
	random_split = np.asarray(random.sample(
		range(0, X_train_All.shape[0]), X_train_All.shape[0]))

	X_train_All = X_train_All[random_split, :, :, :]
	y_train_All = y_train_All[random_split]

	X_valid = X_train_All[10000:15000, :, :, :]
	y_valid = y_train_All[10000:15000]

	X_Pool = X_train_All[20000:60000, :, :, :]
	y_Pool = y_train_All[20000:60000]

	X_train_All = X_train_All[0:10000, :, :, :]
	y_train_All = y_train_All[0:10000]

	# training data to have equal distribution of classes
	idx_0 = np.array(np.where(y_train_All == 0)).T
	idx_0 = idx_0[0:n_init, 0]
	X_0 = X_train_All[idx_0, :, :, :]
	y_0 = y_train_All[idx_0]

	idx_1 = np.array(np.where(y_train_All == 1)).T
	idx_1 = idx_1[0:n_init, 0]
	X_1 = X_train_All[idx_1, :, :, :]
	y_1 = y_train_All[idx_1]

	idx_2 = np.array(np.where(y_train_All == 2)).T
	idx_2 = idx_2[0:n_init, 0]
	X_2 = X_train_All[idx_2, :, :, :]
	y_2 = y_train_All[idx_2]

	idx_3 = np.array(np.where(y_train_All == 3)).T
	idx_3 = idx_3[0:n_init, 0]
	X_3 = X_train_All[idx_3, :, :, :]
	y_3 = y_train_All[idx_3]

	idx_4 = np.array(np.where(y_train_All == 4)).T
	idx_4 = idx_4[0:n_init, 0]
	X_4 = X_train_All[idx_4, :, :, :]
	y_4 = y_train_All[idx_4]

	idx_5 = np.array(np.where(y_train_All == 5)).T
	idx_5 = idx_5[0:n_init, 0]
	X_5 = X_train_All[idx_5, :, :, :]
	y_5 = y_train_All[idx_5]

	idx_6 = np.array(np.where(y_train_All == 6)).T
	idx_6 = idx_6[0:n_init, 0]
	X_6 = X_train_All[idx_6, :, :, :]
	y_6 = y_train_All[idx_6]

	idx_7 = np.array(np.where(y_train_All == 7)).T
	idx_7 = idx_7[0:n_init, 0]
	X_7 = X_train_All[idx_7, :, :, :]
	y_7 = y_train_All[idx_7]

	idx_8 = np.array(np.where(y_train_All == 8)).T
	idx_8 = idx_8[0:n_init, 0]
	X_8 = X_train_All[idx_8, :, :, :]
	y_8 = y_train_All[idx_8]

	idx_9 = np.array(np.where(y_train_All == 9)).T
	idx_9 = idx_9[0:n_init, 0]
	X_9 = X_train_All[idx_9, :, :, :]
	y_9 = y_train_All[idx_9]

	X_train = np.concatenate(
		(X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9), axis=0)
	y_train = np.concatenate(
		(y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis=0)

	print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')

	print('Distribution of Training Classes:', np.bincount(y_train))

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_valid = X_valid.astype('float32')
	X_Pool = X_Pool.astype('float32')
	X_train /= 255
	X_valid /= 255
	X_Pool /= 255
	X_test /= 255

	Y_test = to_categorical(y_test, 10)
	Y_valid = to_categorical(y_valid, 10)
	Y_Pool = to_categorical(y_Pool, 10)
	Y_train = to_categorical(y_train, 10)

	return X_train,Y_train,X_test,Y_test,X_valid,Y_valid,X_Pool,Y_Pool











