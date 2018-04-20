import tensorflow as tf
tf.set_random_seed(1)

def triplet_loss_func(y_true, y_pred, alpha=0.3):
	'''
	Used directly as loss function
		Inputs:
					y_true: True values of classification. (y_train)
					y_pred: predicted values of classification.
					alpha: Distance between positive and negative sample, arbitrarily
						   set to 0.3

		Returns:
					Computed loss

		Function:
					--Implements triplet loss using tensorflow commands
					--The following function follows an implementation of Triplet-Loss 
					  where the loss is applied to the network in the compile statement 
					  as usual.
	'''
	anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

	positive_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), -1)
	negative_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)), -1)

	loss_1 = tf.add(tf.subtract(positive_dist, negative_dist), alpha)
	loss = tf.reduce_sum(tf.maximum(loss_1, 0.0))

	return loss


def triplet_loss_fn(x, alpha=0.3):
	'''
	This is not used in given implementation.
		
	If used, used as the mode of merging.

		Inputs:
					y_true: True values of classification. (y_train)
					y_pred: predicted values of classification.
					alpha: Distance between positive and negative sample, arbitrarily
						   set to 0.3

		Returns:
					Computed loss

		Function:
					--Implements triplet loss using tensorflow commands
					--The following function follows an implementation of Triplet-Loss 
					  where the loss is applied to three separate image-embeddings, in a merge 
					  layer. 
	'''
	anchor, positive, negative = x

	positive_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
	negative_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)), 1)

	loss_1 = tf.add(tf.subtract(positive_dist, negative_dist), alpha)
	loss = tf.reduce_sum(tf.maximum(loss_1, 0.0), 0)

	return loss

