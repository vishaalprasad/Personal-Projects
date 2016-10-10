import numpy as np
import cPickle, gzip

def open_mnist():
	f = gzip.open('mnist.pkl.gz', 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()
	return (train_set, valid_set, test_set)

def create_dataset():
	data = open_mnist()
	train = data[0]; train_data, train_labels = train;
	valid = data[1]; valid_data, valid_labels = valid;
	#no need for validation set in kNN --- not doing parameter tuning
	train_data = np.concatenate((train_data, valid_data))
	train_labels = np.concatenate((train_labels, valid_labels))

	test = data[2]; test_data, test_labels = test;

	#Combine data and labels into one array
	train = np.concatenate((train_data, train_labels[:, None]), axis=1)
	test = np.concatenate((test_data, test_labels[:, None]), axis=1)
	return (train, test)

