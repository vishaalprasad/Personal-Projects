from math import sqrt
from os import path
import numpy as np
import matplotlib.pyplot as plt
import operator
import pickle

import util
import distance_functions
import prototype_dataset


# Helper method

def kNearestNeighbors(train, instance, k, distance_fcn):
	distances = []
	ndim = len(instance)-1

	for i in range(len(train)):
		dist = distance_fcn(train[i], instance)
		temp = train[i]
		distances.append((temp[-1], dist)) #create pair
	distances.sort(key=operator.itemgetter(1)) #sort by 2nd val in pair

	return distances[0:k]

def make_prediction(neighbors):
	votes = {}

	for i in range(len(neighbors)):
		label = neighbors[i][0]
		if label in votes:
			votes[label] += 1
		else:
			votes[label] = 1
	sorted_votes = sorted(votes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sorted_votes[0][0]

#train, test are expected to have class label at the end
def kNN_driver(train, test, k, distance_fcn):
	nErr = 0.0
	count = 0;
	for inst in test:
		neighbors = kNearestNeighbors (train, inst, k, distance_fcn)
		pred = make_prediction(neighbors)
		actual = inst[-1]
		if pred != int(actual):
			nErr += 1
		count += 1
		if count % 10 == 0:
			print count #helps programmer keep sanity

	return 1-nErr/np.shape(test)[0] #accuracy


if __name__ == '__main__':
	####### A section to define hyper-parameters
	k = 1;
	dist_fcn = distance_functions.euclidean_distance;
	dset_sizes = np.multiply([1, 3, 5, 7, 10], 1000);
	prototype_dataset_function = prototype_dataset.random
	prototype_dataset_compression = prototype_dataset.CNN
	####### Create dataset (random will be created inside)
	train, test = util.create_dataset()
    #######

    ####### Other parameters
	num_rand_iters = 10 #how many times should the random algorithm run?
	sem = []
	avgs = []
	pt_avgs = []
    #######
	if not path.isfile('comp_cache.p'):
		compressed_train = prototype_dataset_compression(train) #if the dataset should be compressed
		with open('comp_cache.p', 'wb') as f:
			pickle.dump(compressed_train, f)

	else:
		with open('comp_cache.p', 'r') as f:
			compressed_train = pickle.load(f)

	for dset_size in dset_sizes:
		random_errs = []
		proto_dset = prototype_dataset_function(compressed_train, dset_size)
		pt_avgs.append(kNN_driver(proto_dset, test, k, dist_fcn))
		if not path.isfile('cache.p'):
			for i in range(0,4):
				random_train = prototype_dataset.random(train, dset_size)
				random_errs.append(kNN_driver(random_train, test, k, dist_fcn))

			sem.append(np.asarray(random_errs).std() / sqrt(num_rand_iters))
			avgs.append(np.asarray(random_errs).mean())

	if path.isfile('cache.p'):
		with open('cache.p', 'r') as f:
			sem, avgs = pickle.load(f)
	else:
		with open('cache.p', 'wb') as f:
			pickle.dump([sem, avgs], f)

    ####### Create plot
    # See [1] at the bottom for proper credit 
	line,caps,bars=plt.errorbar(
	    dset_sizes,     # X
	    avgs,    # Y
	    yerr=sem,        # Y-errors
	    fmt="rs--",    # format line like for plot()
	    linewidth=3,   # width of plot line
	    elinewidth=0.5,# width of error bar line
	    #ecolor='k',    # color of error bar
	    capsize=5,     # cap length for error bar
	    capthick=0.5   # cap thickness for error bar
	    )

	plt.setp(line,label="Random")#give label to returned line
	plt.legend(numpoints=1,             	#Set the number of markers in label
	           loc=('upper left'))      	#Set label location
	plt.xlim((0, np.max(dset_sizes)+1000))  #Set X-axis limits
	plt.xticks(dset_sizes)               	#get only ticks we want
	#plt.yticks(dset_sizes)
	line = plt.plot(dset_sizes, pt_avgs)
	plt.setp(line, label="Averaging")
	plt.show()

''' References
[1] http://scientificpythonsnippets.com/index.php/distributions/4-scientific-plotting-with-python-plot-with-error-bars-using-pyplot
'''
