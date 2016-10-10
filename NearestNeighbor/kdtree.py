'''
Credit to:
http://code.activestate.com/recipes/577497-kd-tree-for-nearest-neighbor-search-in-a-k-dimensi/
99% of the code here is taken from the site.
This is used as a quick workbench to test various performance boosters (e.g. PCA)
'''

import collections 
import itertools
import math

Node = collections.namedtuple("Node", 'point axis label left right')

def euclidean_distance(entry1, entry2):
	return np.linalg.norm(entry1-entry2)

class KDTree(object):

	def __init__(self, k, objects=[]):
		def build_tree(objects, axis=0):
			import pdb; pdb.set_trace()

			if not objects:
				return None
			# Step 1) Sort Tree on each dimension
			objects.sort(key=lambda o: o[0][axis])

			# Step 2) Find median
			median_idx = len(objects) // 2
			median_point, median_label = objects[median_idx]

			# Step 3) Find next axis and recurse
			next_axis = (axis + 1) % k
			return Node(median_point, axis, median_label,
						build_tree(objects[:median_idx], next_axis),
						build_tree(objects[median_idx+1:], next_axis))

		self.root = build_tree(list(objects))

	def nearest_neighbor(self, destination):

		best = [None, None, float('inf')]

		def recursive_search(curr):

			if curr is None:
				return
			point, axis, label, left, right = curr

			distance = euclidean_distance(point, destination)
			if distance < best[2]: #new nearest
				best[:] = point, label, distance

			#Find distance and to if  candidates exist in other regions
			diff = destination[axis] - point[axis] 
			close, away = (left, right) if diff <= 0 else (right, left)

			recursive_search(close)
			if diff ** 2 < best[2]:
				recursive_search(away)

		recursive_search(self.root)
		return best[0], best[1], math.sqrt(best[2])

if __name__ == '__main__':
	import util
	train, test = util.create_dataset()
	tree = KDTree(784, train)

	nerr = 0
	for entry in test:
		_, _, mindist = tree.nearest_neighbor(entry)
