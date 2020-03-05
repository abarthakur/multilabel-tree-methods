import copy
import utils
import numpy as np
from sklearn.preprocessing import normalize as normalize_features
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC


class OVAMultiLabelClassifier:

	def __init__(self,base_classifier):
		# base classifier should be like an sklearn classifier
		# importantly support fit(x_mat,)
		self.base_classifier=base_classifier
		self.num_labels=None
		self.classifier_list=None
	
	def fit(self,x_mat,y_mat):
		assert(len(x_mat.shape)==2)
		assert(len(y_mat.shape)==2)
		self.num_labels=y_mat.shape[1]
		self.classifier_list=[None]*self.num_labels
		# get counts of each label
		lcounts=np.sum(y_mat,axis=0)
		self.classifier_list=[None]*self.num_labels
		for lab in range(0,self.num_labels):
			#edge cases
			if lcounts[lab]==0:
				self.classifier_list[lab]="zero"
			elif lcounts[lab]==y_mat.shape[0]:
				self.classifier_list[lab]="one"
			else:
				base=copy.deepcopy(self.base_classifier)
				base.fit(x_mat,y_mat[:,lab].reshape(-1))
				self.classifier_list[lab]=base

	def predict_proba(self,x_tst):
		assert(len(x_tst.shape)==2)
		num_samples=x_tst.shape[0]
		probs=np.zeros((num_samples,self.num_labels))
		for i in range(0,self.num_labels):
			if self.classifier_list[i]=="zero":
				probs[:,i]=np.zeros(num_samples)+0.0
			elif self.classifier_list[i]=="one":
				probs[:,i]=np.zeros(num_samples)+1.0
			else:
				probs[:,i]=self.classifier_list[i].predict_proba(x_tst)[:,1]
		return probs

	def predict(self,x_tst,threshold=0.5,return_probs=False):
		probs=self.predict_proba(x_tst)
		labels=(probs>=threshold)*1
		if return_probs:
			return labels,probs
		else:
			return labels


class BinaryLinearSVCWrapper(LinearSVC):
	'''
	To be used with OVAMultiLabelClassifier ONLY.
	Mocks it for single class case (LinearSVC would throw error)

	predict_proba seems to have been deprecated for LinearSVC, 
	due to (?theoretical) inconsistencies with results of predict.
	Here is a related issue-
		https://github.com/scikit-learn/scikit-learn/issues/13211
	However, for Parabel the following defn of predict_proba is exactly
	what we need.
	'''

	def predict_proba(self,x_tst):
		# edge case : constant classifier
		if self._is_constant_classifier:
			constant_probs=np.zeros((x_tst.shape[0],2))
			constant_probs[:,1]=self._constant_value
			return constant_probs
		# otherwise compute (num_samples,2) array from SVC distances
		# and labels
		distances=self.decision_function(x_tst)
		denoms=1 + np.exp(-1 * distances)
		probs=(1/denoms).reshape(-1,1)
		probs=np.concatenate([(1-probs),probs],axis=1)
		return probs
	
	def predict(self,x_tst,threshold=0.5):
		# threshold only for ?compatibility? purposes
		# edge case : constant classifier
		if self._is_constant_classifier:
 			return self.predict_proba(x_tst)
		else:
			# otherwise, pass SVC predicted labels
			return super().predict(x_tst)

	def fit(self,x_mat,y_mat):
		# must be binary
		assert(len(y_mat.shape)==1)
		# constant cases (act as a constant classifier)
		label_sum=np.sum(y_mat)
		if label_sum==0:
			self._is_constant_classifier=True
			self._constant_value=0
			return
		elif label_sum==y_mat.shape[0]:
			self._is_constant_classifier=True
			self._constant_value=1
			return
		else:
			self._is_constant_classifier=False
		# otherwise fit the SVC
		super().fit(x_mat,y_mat)		


class ClosestMeanClassifier:

	def __init__(self,metric="cosine"):
		assert metric in ["cosine","euclidean"]
		self.metric=metric
		self.means=None

	def fit(self,x_mat,y_mat):
		assert(type(y_mat)==np.ndarray)
		num_labels=y_mat.shape[1]
		feat_dim=x_mat.shape[1]
		means=np.zeros((num_labels,feat_dim))
		for i in range(0,num_labels):
			support_idcs=np.nonzero(y_mat[:,i]>0)[0]
			x_supp=x_mat[support_idcs,:]
			if type(x_supp)==csr_matrix:
				x_supp=x_supp.toarray()
			if self.metric=="cosine":
				x_supp=normalize_features(x_supp)
			means[i,:]=np.mean(x_supp,axis=0)
		if self.metric=="cosine":
			self.means=normalize_features(means)
		elif self.metric=="euclidean":
			self.means=means

	def predict_proba(self,x_tst):
		if type(x_tst)==csr_matrix:
			x_tst=x_tst.toarray()
		if self.metric=="cosine":
			x_tst=normalize_features(x_tst)
			dists=1-utils.calculate_dot_products(x_tst,self.means)
		elif self.metric=="euclidean":
			dists=utils.calculate_euclidean_distances(x_tst,self.means)
		closest=np.argmin(dists,axis=1)
		labels=np.zeros((x_tst.shape[0],self.means.shape[0]))
		labels[np.arange(0,x_tst.shape[0]),closest]=1.0
		return labels

	def predict(self,x_tst,threshold=0.5):
		return self.predict_proba(x_tst)


class MeanLabelVectorClassifier:

	def __init__(self):
		self.y_mean=None

	def fit(self,x_mat,y_mat):
		if type(y_mat)==csr_matrix:
			y_mat=y_mat.toarray()
		# simply calculate the average label vector
		self.y_mean=np.mean(y_mat,axis=0)

	def predict_proba(self,x_tst):
		assert(len(x_tst.shape)==2)
		# not really probabilities, more like soft labels
		# tile the label vector for each sample
		soft_labels=np.tile(self.y_mean,(x_tst.shape[0],1))
		return soft_labels

	def predict(self,x_tst,threshold=0.5):
		# threshold soft labels to get hard labels
		soft_labels=self.predict_proba(x_tst)
		hard_labels=(soft_labels>threshold)*1
		return hard_labels
