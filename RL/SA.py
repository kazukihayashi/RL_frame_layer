import math
import copy
import numpy as np

class INIT_TEMP():

	def __init__(self):
		self.count = 0
		np.random.seed(0)

	def determine_init_temp(self, samples, target_probability=0.8, tol=1.0e-4, p=2.0):
		"""
		samples[n_samples][2]: n_samples transitions. [0]:before [1]:after
		target_probability: This method modifies initial temperature such that initial probability to accept worse transition in SA converges this value.
		tol: absolute tolerance of probability. If tol=0.01 and target_probability=0.5, this method converges with target_probability=(0.49,0.51).
		
		p:
		degree of modification for init_temp. If small, larger modification but less convergence.

		!!! Note that the probability to move to any possible neighbor is uniform, so the target_probability does NOT depend on cost increment !!!

		The detail is described in
		Walid Ben-Ameur; Computing the initial temperature of simulated annealing. Comp.Opt.&App., 29, 369-385, 2004.
		"""
		samples = np.array(samples)
		n_samples, _ = np.shape(samples)

		init_temp = 1000

		while True:
			x = np.sum([np.exp(-np.max(samples[i])/init_temp) for i in range(n_samples)])/np.sum([np.exp(-np.min(samples[i])/init_temp) for i in range(n_samples)])
			print("x=",x)
			if x-target_probability < tol:
				print("init_temp converged.")
				print("init_temp = ",init_temp)
				break
			else:
				init_temp = init_temp*np.power(np.log(x)/np.log(target_probability),1/p)
				self.count += 1

		return init_temp

class SA():
	def __init__(self, init_X, init_F, samples=None):

		if(samples is None):
			self.INIT_T = 10.0
		else:
			self.INIT_T = INIT_TEMP().determine_init_temp(samples)
		self.COOLING_RATE = 0.9
		self.init_X = init_X
		self.init_F = init_F
		self.Reset()

	def Reset(self):

		self.best_X = copy.deepcopy(self.init_X)
		self.best_F = copy.deepcopy(self.init_F)
		self.X_before = copy.deepcopy(self.init_X)
		self.F_before = copy.deepcopy(self.init_F)
		self.temp = copy.deepcopy(self.INIT_T)
		self.count = 0
		self.best_iter = 0
		self.history = [self.init_F]

	def Accept(self, X_after, F_after):

		delta = F_after - self.F_before
		if self.best_F is np.inf:
			self.best_X = copy.deepcopy(X_after)
			self.best_F = copy.deepcopy(F_after)
			self.best_iter = copy.deepcopy(self.count)
			acceptance = True			
		elif F_after < self.best_F:
			self.best_X = copy.deepcopy(X_after)
			self.best_F = copy.deepcopy(F_after)
			self.best_iter = copy.deepcopy(self.count)
			acceptance = True
		elif delta <= 0:
			acceptance = True
		elif np.exp(-delta/self.temp) >= np.random.rand():
			acceptance = True
		else:
			acceptance = False
		return acceptance

	def Select(self, X_after, F_after):

		self.count += 1
		acceptance = self.Accept(X_after, F_after)
		if acceptance==True:
			self.X_before = copy.deepcopy(X_after)
			self.F_before = copy.deepcopy(F_after)
		self.history.append(self.F_before)
		return self.X_before

	def Update_Temp(self):

		self.temp = self.temp * self.COOLING_RATE

