from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from triplet_loss_functions import triplet_loss_func
from keras.optimizers import Adam
import h5py
import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input

def fine_tuned_model(num_classes, shape=(3,224,224)):

	'''
		Inputs:
					shape: The shape of the input tensor. Defaults to the tuple (3,224,224)
					num_classes: The new number of classes for classification. 

		Returns:
					the fine tuned model.

		Functions:
					--Initializes ResNet50 model without the classification layer and replaces
					it with a new classifier with the appropriate number of classes.
					--Freezes the first 15 layers of ResNet50 to avoid overfitting.
					--Reduces learning rate for fine tuning.
					--Compiles with triplet-loss.
					--Any pre-trained model in the keras library can be used with the top off. 
	'''


	initial_model = ResNet50(include_top=False, input_shape=shape)
	last = initial_model.output
	#print(last.shape)
	x = Flatten()(last)
	x = Dense(num_classes, activation='softmax')(x)
	model = Model(initial_model.input, x)

	#Train only higher layers to avoid overfitting
	for layer in model.layers[:15]:
		layer.trainable = False

    #Learning rate is changed to 0.001
	adam = Adam(lr=1e-3, decay=1e-6)
	model.compile(optimizer=adam, loss=triplet_loss_func, metrics=['accuracy'])

	return model



def data_generator(sess,data,labels):
	'''
	This is not used.
		
		Inputs: 
					sess: Instance of the session already initialised in the parent file.
					data: data on which the model has to be trained.
					labels: classes in the classification layer. (y_true)

		Returns:
					Returns the generator function.

		Functions:
					--Resizes and preprocesses images in batches of 16.
					--Yields batches of final images + batches of labels.
					--Used as the generator function in the .fit_generator method.
	'''
	def generator():
		batch_size = 16
		start = 0
		end = start + batch_size
		n = data.shape[0]
		while True:
			batch_of_images_resized = sess.run(tf_resize_op, {batch_of_images_placeholder: data[start:end]})
			batch_of_images_preprocessed = preprocess_input(batch_of_images_resized)
			batch_of_labels = labels[start:end]
            
			start += batch_size
			end   += batch_size
			if start >= n:
				start = 0
				end = batch_size
				yield (batch_of_images_preprocessed, batch_of_labels)
		return generator
