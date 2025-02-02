from __future__ import print_function
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.utils import np_utils, generic_utils
from six.moves import range
import numpy as np
import scipy as sp
from keras import backend as K  
import random
import scipy.io
import matplotlib.pyplot as plt
from keras.regularizers import l2, activity_l2
# from sklearn.metrics import log_loss
# from sklearn.metrics import mean_squared_error
# from math import sqrt



Experiments = 1

batch_size = 128
nb_classes = 10

#use a large number of epochs
nb_epoch = 50

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

score=0
all_accuracy = 0
all_predicted_log_likelihood = 0
all_rmse = 0
acquisition_iterations = 90

#use a large number of dropout iterations
dropout_iterations = 100

Queries = 10


Experiments_All_Accuracy = np.zeros(shape=(acquisition_iterations+1))



for e in range(Experiments):

	print('Experiment Number ', e)

	# the data, shuffled and split between tran and test sets
	(X_train_All, y_train_All), (X_test, y_test) = mnist.load_data()

	X_train_All = X_train_All.reshape(X_train_All.shape[0], 1, img_rows, img_cols)
	X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

	random_split = np.asarray(random.sample(range(0,X_train_All.shape[0]), X_train_All.shape[0]))

	X_train_All = X_train_All[random_split, :, :, :]
	y_train_All = y_train_All[random_split]


	X_valid = X_train_All[10000:15000, :, :, :]
	y_valid = y_train_All[10000:15000]

	X_Pool = X_train_All[20000:60000, :, :, :]
	y_Pool = y_train_All[20000:60000]


	X_train_All = X_train_All[0:10000, :, :, :]
	y_train_All = y_train_All[0:10000]


	#training data to have equal distribution of classes
	idx_0 = np.array( np.where(y_train_All==0)  ).T
	idx_0 = idx_0[0:10,0]
	X_0 = X_train_All[idx_0, :, :, :]
	y_0 = y_train_All[idx_0]

	idx_1 = np.array( np.where(y_train_All==1)  ).T
	idx_1 = idx_1[0:10,0]
	X_1 = X_train_All[idx_1, :, :, :]
	y_1 = y_train_All[idx_1]

	idx_2 = np.array( np.where(y_train_All==2)  ).T
	idx_2 = idx_2[0:10,0]
	X_2 = X_train_All[idx_2, :, :, :]
	y_2 = y_train_All[idx_2]

	idx_3 = np.array( np.where(y_train_All==3)  ).T
	idx_3 = idx_3[0:10,0]
	X_3 = X_train_All[idx_3, :, :, :]
	y_3 = y_train_All[idx_3]

	idx_4 = np.array( np.where(y_train_All==4)  ).T
	idx_4 = idx_4[0:10,0]
	X_4 = X_train_All[idx_4, :, :, :]
	y_4 = y_train_All[idx_4]

	idx_5 = np.array( np.where(y_train_All==5)  ).T
	idx_5 = idx_5[0:10,0]
	X_5 = X_train_All[idx_5, :, :, :]
	y_5 = y_train_All[idx_5]

	idx_6 = np.array( np.where(y_train_All==6)  ).T
	idx_6 = idx_6[0:10,0]
	X_6 = X_train_All[idx_6, :, :, :]
	y_6 = y_train_All[idx_6]

	idx_7 = np.array( np.where(y_train_All==7)  ).T
	idx_7 = idx_7[0:10,0]
	X_7 = X_train_All[idx_7, :, :, :]
	y_7 = y_train_All[idx_7]

	idx_8 = np.array( np.where(y_train_All==8)  ).T
	idx_8 = idx_8[0:10,0]
	X_8 = X_train_All[idx_8, :, :, :]
	y_8 = y_train_All[idx_8]

	idx_9 = np.array( np.where(y_train_All==9)  ).T
	idx_9 = idx_9[0:10,0]
	X_9 = X_train_All[idx_9, :, :, :]
	y_9 = y_train_All[idx_9]

	X_train = np.concatenate((X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9), axis=0 )
	y_train = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis=0 )


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

	Y_test = np_utils.to_categorical(y_test, nb_classes)
	Y_valid = np_utils.to_categorical(y_valid, nb_classes)
	Y_Pool = np_utils.to_categorical(y_Pool, nb_classes)


	#loss values in each experiment
	Pool_Valid_Loss = np.zeros(shape=(nb_epoch, 1)) 	
	Pool_Train_Loss = np.zeros(shape=(nb_epoch, 1)) 
	Pool_Valid_Acc = np.zeros(shape=(nb_epoch, 1)) 	
	Pool_Train_Acc = np.zeros(shape=(nb_epoch, 1)) 
	x_pool_All = np.zeros(shape=(1))

	Y_train = np_utils.to_categorical(y_train, nb_classes)

	print('Training Model Without Acquisitions in Experiment', e)

	model = Sequential()

	model.add(Convolution2D(6, 5, 5, border_mode='valid', input_shape = (1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(16, 5, 5, border_mode='valid', input_shape = (1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape = (1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(64, 5, 5, border_mode='valid', input_shape = (1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(120, 1, 1, border_mode='valid', input_shape = (1, img_rows, img_cols)))
	
	c = 3.5
	Weight_Decay = c / float(X_train.shape[0])
	model.add(Flatten())
	model.add(Dense(84, W_regularizer=l2(Weight_Decay)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))


	model.compile(loss='categorical_crossentropy', optimizer='adam')
	hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_valid, Y_valid))
	Train_Result_Optimizer = hist.history
	Train_Loss = np.asarray(Train_Result_Optimizer.get('loss'))
	Train_Loss = np.array([Train_Loss]).T
	Valid_Loss = np.asarray(Train_Result_Optimizer.get('val_loss'))
	Valid_Loss = np.asarray([Valid_Loss]).T
	Train_Acc = np.asarray(Train_Result_Optimizer.get('acc'))
	Train_Acc = np.array([Train_Acc]).T
	Valid_Acc = np.asarray(Train_Result_Optimizer.get('val_acc'))
	Valid_Acc = np.asarray([Valid_Acc]).T


	Pool_Train_Loss = Train_Loss
	Pool_Valid_Loss = Valid_Loss
	Pool_Train_Acc = Train_Acc
	Pool_Valid_Acc = Valid_Acc


	print('Evaluating Test Accuracy Without Acquisition')
	score, acc = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
	# Y_predict_probabilities = model.predict_proba(X_test, batch_size=batch_size)
	# predicted_log_likelihood = log_loss(Y_test, Y_predict_probabilities)
	# rmse = sqrt(mean_squared_error(Y_test, Y_predict_probabilities))

	# all_predicted_log_likelihood = predicted_log_likelihood
	all_accuracy = acc
	# all_rmse = rmse

	print('Starting Active Learning in Experiment ', e)






	for i in range(acquisition_iterations):
		
		print('POOLING ITERATION', i)

		#take subset of Pool Points for Test Time Dropout 
		#and do acquisition from there
		pool_subset = 2000
		pool_subset_dropout = np.asarray(random.sample(range(0,X_Pool.shape[0]), pool_subset))
		X_Pool_Dropout = X_Pool[pool_subset_dropout, :, :, :]
		y_Pool_Dropout = y_Pool[pool_subset_dropout]

		score_All = np.zeros(shape=(X_Pool_Dropout.shape[0], nb_classes))
		All_Entropy_Dropout = np.zeros(shape=X_Pool_Dropout.shape[0])

		for d in range(dropout_iterations):
			print ('Dropout Iteration', d)
			dropout_score = model.predict_stochastic(X_Pool_Dropout,batch_size=batch_size, verbose=1)
			# np.save('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/active_learning/Acquisition_Functions/Bayesian_Active_Learning/GPU/BALD/Dropout_Scores/'+ 'Experiment_' + str(e)  + '_Dropout_Score_'+str(d)+'.npy',dropout_score)
			#computing G_X
			score_All = score_All + dropout_score

			#computing F_X
			dropout_score_log = np.log2(dropout_score)
			Entropy_Compute = - np.multiply(dropout_score, dropout_score_log)
			Entropy_Per_Dropout = np.sum(Entropy_Compute, axis=1)

			All_Entropy_Dropout = All_Entropy_Dropout + Entropy_Per_Dropout 


		Avg_Pi = np.divide(score_All, dropout_iterations)
		Log_Avg_Pi = np.log2(Avg_Pi)
		Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
		Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

		G_X = Entropy_Average_Pi

		Average_Entropy = np.divide(All_Entropy_Dropout, dropout_iterations)

		F_X = Average_Entropy

		U_X = G_X - F_X

		# THIS FINDS THE MINIMUM INDEX 
		# a_1d = U_X.flatten()
		# x_pool_index = a_1d.argsort()[-Queries:]

		a_1d = U_X.flatten()
		x_pool_index = a_1d.argsort()[-Queries:][::-1]


		#store all the pooled images indexes
		x_pool_All = np.append(x_pool_All, x_pool_index)

		#saving pooled images

		# #save only 3 images per iteration
		# for im in range(x_pool_index[0:2].shape[0]):
		# 	Image = X_Pool[x_pool_index[im], :, :, :]
		# 	img = Image.reshape((28,28))
		#	sp.misc.imsave('/home/ri258/Documents/Project/Active-Learning-Deep-Convolutional-Neural-Networks/ConvNets/Cluster_Experiments/Dropout_Bald/Pooled_Images/' + 'Experiment_' + str(e) + 'Pool_Iter'+str(i)+'_Image_'+str(im)+'.jpg', img)

		Pooled_X = X_Pool_Dropout[x_pool_index, 0:3,0:32,0:32]
		Pooled_Y = y_Pool_Dropout[x_pool_index]	

		#first delete the random subset used for test time dropout from X_Pool
		#Delete the pooled point from this pool set (this random subset)
		#then add back the random pool subset with pooled points deleted back to the X_Pool set
		delete_Pool_X = np.delete(X_Pool, (pool_subset_dropout), axis=0)
		delete_Pool_Y = np.delete(y_Pool, (pool_subset_dropout), axis=0)

		delete_Pool_X_Dropout = np.delete(X_Pool_Dropout, (x_pool_index), axis=0)
		delete_Pool_Y_Dropout = np.delete(y_Pool_Dropout, (x_pool_index), axis=0)

		X_Pool = np.concatenate((X_Pool, X_Pool_Dropout), axis=0)
		y_Pool = np.concatenate((y_Pool, y_Pool_Dropout), axis=0)



		print('Acquised Points added to training set')

		X_train = np.concatenate((X_train, Pooled_X), axis=0)
		y_train = np.concatenate((y_train, Pooled_Y), axis=0)


		# convert class vectors to binary class matrices
		Y_train = np_utils.to_categorical(y_train, nb_classes)

		model = Sequential()

		model.add(Convolution2D(6, 5, 5, border_mode='valid', input_shape = (1, img_rows, img_cols)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
		model.add(Dropout(0.25))

		model.add(Convolution2D(16, 5, 5, border_mode='valid', input_shape = (1, img_rows, img_cols)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
		model.add(Dropout(0.25))

		model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape = (1, img_rows, img_cols)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
		model.add(Dropout(0.25))

		model.add(Convolution2D(64, 5, 5, border_mode='valid', input_shape = (1, img_rows, img_cols)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
		model.add(Dropout(0.25))

		model.add(Convolution2D(120, 1, 1, border_mode='valid', input_shape = (1, img_rows, img_cols)))
		
		c = 3.5
		Weight_Decay = c / float(X_train.shape[0])
		model.add(Flatten())
		model.add(Dense(84, W_regularizer=l2(Weight_Decay)))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(nb_classes))
		model.add(Activation('softmax'))

		model.compile(loss='categorical_crossentropy', optimizer='adam')
		hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_valid, Y_valid))

		Train_Result_Optimizer = hist.history
		Train_Loss = np.asarray(Train_Result_Optimizer.get('loss'))
		Train_Loss = np.array([Train_Loss]).T
		Valid_Loss = np.asarray(Train_Result_Optimizer.get('val_loss'))
		Valid_Loss = np.asarray([Valid_Loss]).T
		Train_Acc = np.asarray(Train_Result_Optimizer.get('acc'))
		Train_Acc = np.array([Train_Acc]).T
		Valid_Acc = np.asarray(Train_Result_Optimizer.get('val_acc'))
		Valid_Acc = np.asarray([Valid_Acc]).T
		

		#Accumulate the training and validation/test loss after every pooling iteration - for plotting
		Pool_Valid_Loss = np.append(Pool_Valid_Loss, Valid_Loss, axis=1)
		Pool_Train_Loss = np.append(Pool_Train_Loss, Train_Loss, axis=1)
		Pool_Valid_Acc = np.append(Pool_Valid_Acc, Valid_Acc, axis=1)
		Pool_Train_Acc = np.append(Pool_Train_Acc, Train_Acc, axis=1)	

		print('Evaluate Model Test Accuracy with pooled points')

		score, acc = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
		print('Test accuracy:', acc)
		all_accuracy = np.append(all_accuracy, acc)
		# Y_predict_probabilities = model.predict_proba(X_test, batch_size=batch_size)
		# predicted_log_likelihood = log_loss(Y_test, Y_predict_probabilities)
		# rmse = sqrt(mean_squared_error(Y_test, Y_predict_probabilities))

		# all_predicted_log_likelihood = np.append(all_predicted_log_likelihood, predicted_log_likelihood)
		# all_rmse = np.append(all_rmse, rmse)

		print('Use this trained model with pooled points for Dropout again')


	print('Storing Accuracy Values over experiments')
	Experiments_All_Accuracy = Experiments_All_Accuracy + all_accuracy


	print('Saving Results Per Experiment')
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Model_Architectures/Results/'+'LetNet_Higher_All_Layers_Bald_Q10_N1000_Train_Loss_'+ 'Experiment_' + str(e) + '.npy', Pool_Train_Loss)
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Model_Architectures/Results/'+ 'LetNet_Higher_All_Layers_Bald_Q10_N1000_Valid_Loss_'+ 'Experiment_' + str(e) + '.npy', Pool_Valid_Loss)
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Model_Architectures/Results/'+'LetNet_Higher_All_Layers_Bald_Q10_N1000_Train_Acc_'+ 'Experiment_' + str(e) + '.npy', Pool_Train_Acc)
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Model_Architectures/Results/'+ 'LetNet_Higher_All_Layers_Bald_Q10_N1000_Valid_Acc_'+ 'Experiment_' + str(e) + '.npy', Pool_Valid_Acc)
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Model_Architectures/Results/'+'LetNet_Higher_All_Layers_Bald_Q10_N1000_Pooled_Image_Index_'+ 'Experiment_' + str(e) + '.npy', x_pool_All)
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Model_Architectures/Results/'+ 'LetNet_Higher_All_Layers_Bald_Q10_N1000_Accuracy_Results_'+ 'Experiment_' + str(e) + '.npy', all_accuracy)


print('Saving Average Accuracy Over Experiments')

Average_Accuracy = np.divide(Experiments_All_Accuracy, Experiments)

np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Model_Architectures/Results/'+'LetNet_Higher_All_Layers_Bald_Q10_N1000_Average_Accuracy'+'.npy', Average_Accuracy)












