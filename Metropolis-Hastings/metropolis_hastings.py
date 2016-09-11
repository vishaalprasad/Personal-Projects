'''
Metropolis-Hastings: an MCMC method when you can't sample directly from the
dataset. Why not rejection sampling or something similar? When the distribution
is high-dimensional, the acceptance ratio falls low. Or we can't (for whatever
reason) can't sample IID. We need another method that isn't beholden to IID. 

This can be done via Markov Chains, where we choose correlated data so that
we stay in high-probability areas. Enter Metropolis-Hastings.

Acknowledgement to http://mlwhiz.com/blog/2015/08/19/MCMC_Algorithms_Beta_Distribution/
which helped especially with the visualization. 
'''
import numpy as np
import random
import math
import matplotlib.pyplot as plt 
import matplotlib.mlab as mlab 
import scipy.stats 

#First, define useful helper methods.

#Gaussian method: this will be used for what we are sampling
def gauss_pdf(x, mean=0,stddev=1.0):
	return math.exp(-(x-mean)**2/(2*stddev**2)) #don't worry about norm const!

#Given some probabliity p, used to determine whether to accept or reject.
def standard_uniform(p):
	if p >= random.uniform(0,1):
		return True
	else:
		return False

'''
Actual Metropolis Hastings algorithm.
Hard-coded so the function we are sampling from is standard Gaussian.
Hard-coded so the proposal distribution is zero-centered Gaussian with variable sigma. 

Param List:
Initial: The initial guess. Arbitrarily let 0 be default.
Iterations: The total number of iterations M-H runs for. 
Count: The count of actually used iterations; the last "count" iterations.
Note that Count < Iterations necessarily.
True mean, sigma: The parameters for our true model that we are sampling from.


'''
def gaussian_metropolis_hastings(initial=0, iterations=10000, count=5000, proposal_mean=0, 
				 proposal_sigma=1.0, true_mean=0, true_sigma=1.0):

	states = []
	states.append(initial)
	curr = initial
	for i in range (0, iterations):
		proposal = scipy.stats.norm.rvs(proposal_mean, proposal_sigma) #gives us a proposal based on the proposal dist
		ap = min(1, gauss_pdf(proposal, true_mean, true_sigma)/ #acceptance probability
			gauss_pdf(curr, true_mean, true_sigma))
		if standard_uniform(ap):
			curr = proposal
		states.append(curr)

	return states[-count:] #return the last "count" entries

# Visualize the gaussian metropolis hastings
def visualize_gaussian_metropolis_hastings_estimate(proposal_sigma, proposal_mean, mean, sigma):
	ys = []
	xs = []
	i_list = np.mgrid[-5:5:200j] # This will allow us to model the pdf of any distribution directly via discretization
	for i in i_list:
		xs.append(i)
		ys.append(scipy.stats.norm(mean, sigma).pdf(i))
	plt.plot(xs, ys, label='True Gaussian Distribution: mean='+str(mean)+
		', sigma='+str(sigma))
	approx = gaussian_metropolis_hastings(iterations = 1000000, count = 800000, proposal_sigma = 5, true_mean = mean, true_sigma = sigma)
	plt.hist(approx, normed=True, bins=100, histtype='step', label='Metropolis-Hastings Approximation')
	plt.legend()
	plt.show()


if __name__ == '__main__':
	visualize_gaussian_metropolis_hastings_estimate(0, 5.0, 0, 1.0)
