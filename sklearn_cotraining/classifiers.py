import numpy as np
import random
import copy
class CoTrainingClassifier(object):
	"""
	Parameters:
	clf - The classifier that will be used in the cotraining algorithm on the X1 feature set
		(Note a copy of clf will be used on the X2 feature set if clf2 is not specified).

	clf2 - (Optional) A different classifier type can be specified to be used on the X2 feature set
		 if desired.

	p - (Optional) The number of positive examples that will be 'labeled' by each classifier during each iteration
		The default is the is determined by the smallest integer ratio of positive to negative samples in L (from paper)

	n - (Optional) The number of negative examples that will be 'labeled' by each classifier during each iteration
		The default is the is determined by the smallest integer ratio of positive to negative samples in L (from paper)

	k - (Optional) The number of iterations
		The default is 30 (from paper)

	u - (Optional) The size of the pool of unlabeled samples from which the classifier can choose
		Default - 75 (from paper)
	"""

	def __init__(self, clf, clf2=None, p=-1, n=-1, k=30, u = 75):
		self.clf1_ = clf
		
		#we will just use a copy of clf (the same kind of classifier) if clf2 is not specified
		if clf2 == None:
			self.clf2_ = copy.copy(clf)
		else:
			self.clf2_ = clf2

		#if they only specify one of n or p, through an exception
		if (p == -1 and n != -1) or (p != -1 and n == -1):
			raise ValueError('Current implementation supports either both p and n being specified, or neither')

		self.p_ = p
		self.n_ = n
		self.k_ = k
		self.u_ = u

		random.seed()


	def fit(self, X1, X2, y):
		"""
		Description:
		fits the classifiers on the partially labeled data, y.

		Parameters:
		X1 - array-like (n_samples, n_features_1): first set of features for samples
		X2 - array-like (n_samples, n_features_2): second set of features for samples
		y - array-like (n_samples): labels for samples, -1 indicates unlabeled

		"""

		#we need y to be a numpy array so we can do more complex slicing
		y = np.asarray(y)

		#set the n and p parameters if we need to
		if self.p_ == -1 and self.n_ == -1:
			num_pos = sum(1 for y_i in y if y_i == 1)
			num_neg = sum(1 for y_i in y if y_i == 0)
			
			n_p_ratio = num_neg / float(num_pos)
		
			if n_p_ratio > 1:
				self.p_ = 1
				self.n_ = round(self.p_*n_p_ratio)

			else:
				self.n_ = 1
				self.p_ = round(self.n_/n_p_ratio)

		assert(self.p_ > 0 and self.n_ > 0 and self.k_ > 0 and self.u_ > 0)

		#the set of unlabeled samples
		U = [i for i, y_i in enumerate(y) if y_i == -1]

		#we randomize here, and then just take from the back so we don't have to sample every time
		random.shuffle(U)

		#this is U' in paper
		U_ = U[-min(len(U), self.u_):]

		#the samples that are initially labeled
		L = [i for i, y_i in enumerate(y) if y_i != -1]

		#remove the samples in U_ from U
		U = U[:-len(U_)]


		it = 0 #number of cotraining iterations we've done so far

		#loop until we have assigned labels to everything in U or we hit our iteration break condition
		while it != self.k_ and U:
			it += 1

			self.clf1_.fit(X1[L], y[L])
			self.clf2_.fit(X2[L], y[L])

			y1 = self.clf1_.predict(X1[U_])
			y2 = self.clf2_.predict(X2[U_])

			n, p = [], []
			
			for i, (y1_i, y2_i) in enumerate(zip(y1, y2)):
				#we added all that we needed to for this iteration, so break
				if len(p) == 2 * self.p_ and len(n) == 2 * self.n_:
					break

				#update our newly 'labeled' samples.  Note that we are only 'labeling' a single sample
				#with each inner iteration.  We want to add 2p + 2n samples per outer iteration, but classifiers must agree

				if y1_i == y2_i == 1 and len(p) < self.p_:
					p.append(i)

				if y2_i == y1_i == 0 and len(n) < self.n_:
					n.append(i)


			#label the samples and remove thes newly added samples from U_
			y[[U_[x] for x in p]] = 1
			y[[U_[x] for x in n]] = 0

			L.extend([U_[x] for x in p])
			L.extend([U_[x] for x in n])

			#TODO: optimize these removals from U_
			#this is currently (2p + 2n)O(n)
			#and I think it can be reduced to O(n) rather easily
			for i in p: U_.pop(i)
			for i in n: U_.pop(i)

			#add new elements to U_
			add_counter = 0 #number we have added from U to U_
			num_to_add = len(p) + len(n)
			while add_counter != num_to_add and U:
				add_counter += 1
				U_.append(U.pop())


			#TODO: Handle the case where the classifiers fail to agree on any of the samples (i.e. both n and p are empty)


		#let's fit our final model
		self.clf1_.fit(X1[L], y[L])
		self.clf2_.fit(X2[L], y[L])


	#TODO: Move this outside of the class into a util file.
	def supports_proba(self, clf, x):
		"""Checks if a given classifier supports the 'predict_proba' method, given a single vector x"""
		try:
			clf.predict_proba([x])
			return True
		except:
			return False
	
	def predict(self, X1, X2):
		"""
		Predict the classes of the samples represented by the features in X1 and X2.

		Parameters:
		X1 - array-like (n_samples, n_features1)
		X2 - array-like (n_samples, n_features2)

		
		Output:
		y - array-like (n_samples)
			These are the predicted classes of each of the samples.  If the two classifiers, don't agree, we try
			to use predict_proba and take the classifier with the highest confidence and if predict_proba is not implemented, then we randomly
			assign either 0 or 1.  We hope to improve this in future releases.

		"""

		y1 = self.clf1_.predict(X1)
		y2 = self.clf2_.predict(X2)

		proba_supported = self.supports_proba(self.clf1_, X1[0]) and self.supports_proba(self.clf2_, X2[0])

		#fill y_pred with -1 so we can identify the samples in which the classifiers failed to agree
		y_pred = np.asarray([-1] * X1.shape[0])

		for i, (y1_i, y2_i) in enumerate(zip(y1, y2)):
			if y1_i == y2_i:
				y_pred[i] = y1_i
			elif proba_supported:
				y1_probs = self.clf1_.predict_proba([X1[i]])[0]
				y2_probs = self.clf2_.predict_proba([X2[i]])[0]
				sum_y_probs = [prob1 + prob2 for (prob1, prob2) in zip(y1_probs, y2_probs)]
				max_sum_prob = max(sum_y_probs)
				y_pred[i] = sum_y_probs.index(max_sum_prob)

			else:
				#the classifiers disagree and don't support probability, so we guess
				y_pred[i] = random.randint(0, 1)

			
		#check that we did everything right
		assert not (-1 in y_pred)

		return y_pred


	def predict_proba(self, X1, X2):
		"""Predict the probability of the samples belonging to each class."""
		y_proba = np.full((X1.shape[0], 2), -1)

		y1_proba = self.clf1_.predict_proba(X1)
		y2_proba = self.clf2_.predict_proba(X2)

		for i, (y1_i_dist, y2_i_dist) in enumerate(zip(y1_proba, y2_proba)):
			y_proba[i][0] = (y1_i_dist[0] + y2_i_dist[0]) / 2
			y_proba[i][1] = (y1_i_dist[1] + y2_i_dist[1]) / 2

		_epsilon = 0.0001
		assert all(abs(sum(y_dist) - 1) <= _epsilon for y_dist in y_proba)
		return y_proba
