#Nearest neighbor can have all sorts of distance functions. The basic is the
#2-norm, AKA Euclidean. This file defines a range of distance functions to be
#used
import numpy as np

def euclidean_distance(entry1, entry2):
	return np.linalg.norm(entry1-entry2)
