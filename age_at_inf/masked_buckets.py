""" masked_buckets.py

A soft-max classification smoother over time, but with additional flexibility to account for
sampling biases (like certain categories being unobservable because of the way the data is
collected). """
import sys

## Standard imports 
import numpy as np
import matplotlib.pyplot as plt

## For evaluating the posterior and
## fitting the model.
from scipy.optimize import minimize

def masked_vector_soft_max(theta,mask):
	a = np.exp(theta)
	a = a*mask
	return a/(a.sum(axis=1)[:,np.newaxis])

class BinomialPosterior(object):

	""" This class encapsulates the log posterior, it's derivative, and associated
	helper functions, to be passed to scipy.optimize. Likelihood is assumed to be binomial, 
	i.e. without any additional over-dispersion. """

	def __init__(self,h_gt,correlation_time=4.34524,g2g_correlation=0,mask=None):

		## Get the problem geometry from
		## the observed h_gt dataset.
		self.T, self.G = h_gt.shape

		## Store the data, including the total
		## observations in each time step and the number
		## of observations not in that category.
		self.h_gt = h_gt
		self.H_t = h_gt.sum(axis=1)
		self.h_tilde_gt = self.H_t.values[:,np.newaxis] - h_gt

		## Construct the prior correlation matrix
		D2 = np.diag(self.T*[-2])+np.diag((self.T-1)*[1],k=1)+np.diag((self.T-1)*[1],k=-1)
		D2[0,2] = 1
		D2[-1,-3] = 1
		self.lam = np.dot(D2.T,D2)*((correlation_time**4)/8.)
		self.correlation_time = correlation_time

		## group-to-group smoothing is also possible, and requires
		## a periodic ridge regression contribution.
		D2 = np.diag(self.G*[-2])+np.diag((self.G-1)*[1],k=1)+np.diag((self.G-1)*[1],k=-1)
		#D2[0,-1] = 1 ## Periodic BCs
		#D2[-1,0] = 1
		D2[0,2] = 1
		D2[-1,-3] = 1
		self.g2g = np.dot(D2.T,D2)*((g2g_correlation**4)/8.)
		self.g2g_correlation = g2g_correlation

		## Finally, construst the mask if needed. If the mask is None, all
		## catagories are considered possible at all times.
		if mask is None:
			self.mask = np.ones(self.h_gt.shape)
		else:
			assert np.isin(mask,[0,1]).all(), \
				   "The mask must be entirely 1s and 0s!"
			self.mask = mask

	def __call__(self,theta):

		""" Evaluate the log posterior with the final group as
		a hard-coded reference for uniqueness purposes. """

		## Transform and compute group prevalence
		## over time
		theta = theta.reshape((self.T,self.G-1))
		theta = np.hstack([theta,np.zeros((self.T,1))])
		p_gt = masked_vector_soft_max(theta,self.mask)
		
		## Compute the log_prior
		neg_log_prior = np.trace(np.dot(theta.T,np.dot(self.lam,theta)))

		## Add the group to group component as well.
		neg_log_prior += np.trace(np.dot(theta,np.dot(self.g2g,theta.T)))
		
		## Compute log likelihood, which is a cross entropy 
		## in each group-time-pair. Since you can now have masked group-times,
		## some probabilities are definitely 0. For efficiency, we compute them anyway,
		## and then correct them.
		with np.errstate(divide="ignore",invalid="ignore"):
			ll = self.h_gt.values*np.log(p_gt)+(self.h_tilde_gt.values)*np.log(1.-p_gt)
		ll[np.isnan(ll)] = 0
		
		return ll.sum()-neg_log_prior

	def gradient(self,theta):

		""" Compute the gradient, again hard-coded to use the final group as 
		a reference for uniqueness reasons. """

		## Transform and compute group prevalence
		## over time
		theta = theta.reshape((self.T,self.G-1))
		theta = np.hstack([theta,np.zeros((self.T,1))])
		p_gt = masked_vector_soft_max(theta,self.mask)

		## Compute LL components, again correcting for 0 and 1 issues
		with np.errstate(divide="ignore",invalid="ignore"):
			dll = (self.h_gt.values/p_gt)-(self.h_tilde_gt.values/(1.-p_gt))
		dll[np.isnan(dll)] = 0

		## Seperate into diagonal and off-diagonal pieces
		diagonal_term = dll*p_gt
		off_diagonal_term = -((dll*p_gt).sum(axis=1)[:,np.newaxis])*p_gt

		## Compute the prior component, first correlation over time, then
		## correlation accross groups.
		prior = 2.*np.dot(self.lam,theta)
		prior += 2.*np.dot(theta,self.g2g)

		## Combine and reshape
		jacobian = diagonal_term + off_diagonal_term - prior
		jacobian = jacobian[:,:-1].reshape(-1)

		return jacobian

def FitModel(log_post,**kwargs):

	""" A wrapper function that uses BFGS in scipy to fit the model. """

	## Set up the initial guess using the fraction associated with
	## each group in the data.
	theta0 = log_post.h_gt.values/(log_post.H_t.values[:,np.newaxis])
	theta0 = theta0[:,:-1].reshape(-1)

	## Use scipy's BFGS implementation to maximize the log-posterior.
	result = minimize(lambda x: -log_post(x),
					  x0 = theta0,
					  method = "BFGS",
					  jac = lambda x: -log_post.gradient(x),
					  **kwargs,
					  )
	return result

def SampleBuckets(result,log_post,N=10000):

	""" Sample the bucket distribution given a results class output from
	scipy.minimize. This is for uncertainty propogation later on, and has a specific
	function here since sampling w.r.t. a specified reference group (the last group in this
	case) requires a little extra care. """

	## Unpack and sample
	theta, theta_cov = result["x"], result["hess_inv"]
	theta_samples = np.random.multivariate_normal(theta,theta_cov,size=(N,))
	theta_samples = theta_samples.reshape((N,log_post.T,log_post.G-1))
	
	## Add the reference group
	theta_samples = np.dstack([theta_samples,np.zeros((N,log_post.T,1))])
	
	## Take the soft-max and return
	a_samples = np.exp(theta_samples)
	a_samples = a_samples/(a_samples.sum(axis=-1)[:,:,np.newaxis])
	return a_samples