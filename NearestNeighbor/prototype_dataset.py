#This file is to create a new dataset to be used for a KNN algorithm using
#various types of algorithms. 
import numpy as np

# The most basic type of prototype selection is random.
def random(train_set, nSamples):
	if len(train_set) <= nSamples: return train_set
	
	train_set = np.asarray(train_set) #just make sure it's a numpy array
	shuffled = np.random.permutation(train_set)
	return shuffled[0:nSamples, :]

''' The most basic type of prototype generation is to simply average classes together
 ASSUMPTIONS MADE:
  1) The last column of the 2nd dimension of train_set is the labels
  2) Class sizes are even! This can easily be generalized so this isn't the case
     but this assumption allows for quick prototyping on MNIST.
'''

def class_averaging(train_set, nSamples):
	train_set = np.asarray(train_set) #just make sure it's a numpy array
	shape = np.shape(train_set)
	dset_size = shape[0]
	img_size = shape[1]
	train_set = train_set[train_set[:,img_size-1].argsort()]

	#Lazy code: when chunking dataset by class labels, you might find 'chunks' that
	#are mixed. Just toss out the ones that don't match. the label.
	#TODO: Change the code to calculate stride based on class type. Should be easy.
	stride = dset_size/nSamples
	initial = 0
	averaged_dataset = np.zeros((nSamples, img_size))
	for i in range(0, nSamples):
		if train_set[initial, img_size-1] == train_set[initial+stride-1, img_size-1]:
			subset = train_set[initial:initial+stride, :]
		else: #this subsample doesn't have all the same. 
			cur_idx = initial 
			while cur_idx != dset_size and \
				train_set[cur_idx, img_size-1] == train_set[cur_idx+1, img_size-1]:
				cur_idx = cur_idx+1

			#cur_idx should now be the index of the last one in the class.	
			subset = train_set[initial:cur_idx, :]
		if len(subset) == 0:
			continue
		averaged_dataset[i] = sum(subset)/len(subset) #take the average
		initial = initial + stride
	return averaged_dataset
'''
A simple idea: Go through the training set and return a random subset. However,
look through the scheme running 3-NN on the entire dataset and remove any that fail.
Add in however many were removed randomly. Do until convergence, or run out of elements.
'''  

def randomized_editing(train_set, nSamples):
	from sklearn.neighbors import KNeighborsClassifier
	shape = np.shape(train_set)
	img_size = shape[1]
	X = random(train_set, nSamples)
	neigh = KNeighborsClassifier(n_neighbors=3, algorithm='auto',)
	neigh.fit(train_set[:, 0:img_size-2], train_set[:, img_size-1]) 
	predictions = neigh.predict(X[:,0:img_size-2])
	correct = predictions==X[:,img_size-1]
	count = 1
	for i in range(0, len(correct)):
		if not correct[i]: #mispredicted here? goodbye example
			X[i,:] = train_set[nSamples+count,:]
			count += 1
	return X

'''
Most online algorithms will give a percent reduction. So use that concept and then
use randomized_editing to select which ones to use, if necessary.

If prototype dataset < nSamples, we can just duplicate entries, which allows 1-NN
to not do any worse. In practice, we'll simply return a smaller dataset size.
'''
def CNN(train_set):
	from sklearn.neighbors import KNeighborsClassifier
	train_labs = train_set[:, 784]
	train_set = train_set[:, 0:783]
	S,G = [train_set[0]],[] #store, grab_bag
	S_labs, G_labs = [train_labs[0]], []
	neigh = KNeighborsClassifier(n_neighbors=1, algorithm='auto')
	neigh.fit(S, S_labs)

	for i in xrange(1,len(train_set)):
		curr = train_set[i]
		curr_lab = train_labs[i]
		prediction = neigh.predict(curr.reshape(1, -1))
		if prediction == curr_lab:
			G.append(curr); G_labs.append(curr_lab)
		else:
			S.append(curr); S_labs.append(curr_lab)
			neigh.fit(S, S_labs)


	condition = True
	neigh.fit(S, S_labs)

	while condition:
		if G == []:  break; 
		for i in xrange(1, len(G)):
			curr_elem = G[i]; curr_lab = G_labs[i]
			prediction = neigh.predict(curr.reshape(1, -1))
			condition = False;
			import pdb; pdb.set_trace()
			if prediction == curr_lab: 
				elem = G.pop(i); lab = G_labs.pop(i)
				S.append(elem); S_labs.append(lab)
				neigh.fit(S, S_labs)
				condition = True; break;
	print 
	return S






