from fine_tune_model import fine_tuned_model, data_generator
from keras.datasets import cifar10
import numpy as np
import tensorflow as tf
from keras import backend as K

'''
			Note:
				I've used Cifar 10 dataset because it is already available in Keras.

				Cifar-10 is a dataset of 60,000 images of size 32x32. 
				CNN models like ResNet50 / AlexNet / VGG16 etc. accept images of
				size = 224x224 as input. 
				
				Therefore Cifar-10 is not a suitable dataset to work with in 
				this particular example, although padding may help.

				Some sections of code have been commented in the following file.
				They showcase a work-around to the above problem, 
				but threw an unexpected error at runtime.
			
			data_format = 'channels_first'
'''

if __name__ == '__main__':
	img_rows, img_cols = 224, 224
	channels = 3
	num_classes = 10 
	batch_size = 16 
	epochs = 10

	#Replace with your own dataset.
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	
	#Size of training batch
	batch_train = x_train.shape[0]
	#Size of testing batch
	batch_test = x_test.shape[0]

	y_train = y_train.flatten()
	y_test  = y_test.flatten()

	#Uncomment this to reshape images to a size acceptable by the CNN
	'''
	x_train = x_train.reshape(batch_size, channels, img_rows, img_cols)
	x_test = x_test.reshape(batch_size, channels, img_rows, img_cols)
	'''

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	#-----------------------------------------------------------------------------
	#Initialized tensorflow session.
	'''
	sess = tf.InteractiveSession()
	K.set_session(sess)
	'''
	#-----------------------------------------------------------------------------

	#10 classes because dataset is Cifar-10
	model = fine_tuned_model(10, (channels,img_rows,img_cols))
	
	#-----------------------------------------------------------------------------
	#fit_generator is a method that fits the model on data that is 
	#processed in batches before being sent to the model 
	'''
	#create one-hot encodings of the true image classes
	y_train_one_hot = tf.one_hot(y_train, num_classes).eval()
	data_train_gen = data_generator(sess, x_train, y_train_one_hot)
    # Fit model on data using fit_generator
	model.fit_generator(data_train_gen(), epochs=batch_train/batch_size, verbose=1)
	'''
	#-------------------------------------------------------------------------------
	
	#Fit model on the training and testing datasets
	model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              verbose=1,
              validation_data=(x_test, y_test)
              )

    # Make predictions
	predictions_valid = model.predict(x_test, batch_size=batch_size, verbose=1)
