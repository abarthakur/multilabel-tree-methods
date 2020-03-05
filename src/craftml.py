import copy
from clustertree import ClusterTree
from classifiers import ClosestMeanClassifier,MeanLabelVectorClassifier
from partition import SphericalKMeansPartitioner
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils import murmurhash3_32 as m3hash


class HashingTrickProjector:

	def __init__(self, input_dim, output_dim,batch_size=1000):
		self.num_seeds=2
		self.input_dim=input_dim
		self.output_dim=output_dim
		self.batch_size=batch_size
		# set seeds
		self.index_seed=np.random.randint(0,np.iinfo(np.int32).max,dtype='int32')
		self.sign_seed=np.random.randint(0,np.iinfo(np.int32).max,dtype='int32')
		# set projection matrix

	def _get_projection_matrix(self):
		proj_matrix=np.zeros((self.input_dim,self.output_dim))
		for idx in range(0,self.input_dim):
			# get h(idx) -> this hash gives us which output index
			index_hash = m3hash(idx,seed=self.index_seed,positive=True)
			index_hash = index_hash % self.output_dim
			# get E(idx) -> this hash gives us the sign
			sign_hash = m3hash(idx,seed=self.sign_seed,positive=True)
			sign_hash = sign_hash % 2
			sign = 1 if sign_hash==0 else -1
			# set the h(idx) column of the idx row to sign
			proj_matrix[idx,index_hash]=sign
		return proj_matrix    

	def project_data(self,x_mat):
		assert(x_mat.shape[1]==self.input_dim)
		# sparse matrix encountered
		if type(x_mat)==csr_matrix:
			return self._project_sparse_data(x_mat)
		# else for a dense array
		return self._project_dense_data(x_mat)

	def _project_dense_data(self,x_mat):
		num_samples=x_mat.shape[0]
		# compute index and sign map for each index of input_arr
		index_map=np.zeros(self.input_dim)
		sign_map=np.zeros(self.input_dim)
		for idx in range(0,self.input_dim):
			# calculate index hash 
			index_hash=m3hash(idx,seed=self.index_seed,positive=True)
			output_idx=index_hash % self.output_dim
			index_map[idx]=output_idx
			# calculate sign hash
			sign_hash=m3hash(idx,seed=self.sign_seed,positive=True)
			sign_hash=sign_hash % 2
			sign = 1 if sign_hash==0 else -1
			sign_map[idx]=sign

		# compute output matrix
		output_matrix=np.zeros((num_samples,self.output_dim))
		for j in range(0,self.output_dim):
			hashed_idcs=index_map==j
			output_matrix[:,j]=np.sum(x_mat[:,hashed_idcs] * sign_map[hashed_idcs],axis=1)
		return output_matrix

	def _project_sparse_data(self,x_mat):
		num_samples=x_mat.shape[0]
		# compute output sparse matrix in batches
		output_matrix=csr_matrix((num_samples,self.output_dim))
		batch_size=self.batch_size
		num_batches=int(np.ceil(num_samples/batch_size))
		start=0
		for i in range(0,num_batches):
			end=np.minimum(start+batch_size,num_samples)
			# get slice
			x_slice=x_mat[start:end,:]
			# project slice
			proj_slice=self._project_dense_data(x_slice.toarray())
			proj_slice=csr_matrix(proj_slice)
			# set projected slice
			output_matrix[start:end,:]=proj_slice
			start+=batch_size
		return output_matrix


class CraftMLStoppingCondition:

	def __init__(self,min_leaf_size=10,max_depth=800):
		assert(min_leaf_size>0)
		self.min_leaf_size=min_leaf_size
		self.max_depth=max_depth
	
	def check(self,cnode,x_mat,y_mat,repre):
		# reached maximum leaf size
		if len(cnode.train_idcs) <= self.min_leaf_size:
			return True
		if cnode.depth >= self.max_depth:
			return True
		# degenerate features,labels or clustering representations
		check_degenerate=[x_mat,y_mat,repre]
		for arr in check_degenerate:
			mat_supp=arr[cnode.train_idcs,:]
			if type(mat_supp)==csr_matrix:
				mat_supp=mat_supp.toarray()
			if np.allclose(mat_supp,mat_supp[0,:]):
				return True	
		return False



class CraftMLEnsemble:

	def __init__(self,num_trees=50,projector=HashingTrickProjector,
				partitioner=SphericalKMeansPartitioner(10),
				leaf_classifier=MeanLabelVectorClassifier(),
				internal_classifier=ClosestMeanClassifier("cosine"),
				stopping_condition=CraftMLStoppingCondition(),
				max_proj_dim=10000):
		
		self.static_tree_args={
			"partitioner":partitioner,
			"leaf_classifier":leaf_classifier,
			"internal_classifier":internal_classifier,
			"stopping_condition":stopping_condition
		}

		self.projector_template=projector
		self.max_proj_dim=max_proj_dim
		self.num_trees=num_trees
		self.trees=[]
		self.projectors_x=[]
		self.projectors_y=[]

	def fit(self,x_mat,y_mat):
		self.num_features=x_mat.shape[1]
		self.num_labels=y_mat.shape[1]
		self.num_points=x_mat.shape[0]
		self._initialize_projectors()
		for i in range(0,self.num_trees):
			print("Fitting Tree #",i)
			ctree = ClusterTree(**copy.deepcopy(self.static_tree_args))
			proj_x=self.projectors_x[i].project_data(x_mat)
			proj_y=self.projectors_y[i].project_data(y_mat)
			ctree.fit(proj_x,y_mat,proj_y)
			self.trees.append(ctree)

	def _initialize_projectors(self):
		# initialize random projectors for each tree
		# the projections are the (?main) source of diversity in the ensemble
		self.projector_x_dim=np.minimum(self.num_features,self.max_proj_dim)
		self.projector_y_dim=np.minimum(self.num_labels,self.max_proj_dim)
		for i in range(0,self.num_trees):
			proj_x = self.projector_template(input_dim=self.num_features,output_dim=self.projector_x_dim)
			proj_y = self.projector_template(input_dim=self.num_labels,output_dim=self.projector_y_dim)
			self.projectors_x.append(proj_x)
			self.projectors_y.append(proj_y)

	def predict_proba(self,x_tst):
		probs=None
		for i in range(0,self.num_trees):
			print("Predicting on Tree #",i)
			proj_x_tst=self.projectors_x[i].project_data(x_tst)
			probs_t=self.trees[i].predict_proba(proj_x_tst)
			if probs is None:
				probs=probs_t
			else:
				probs+=probs_t
		probs=probs/float(self.num_trees)
		return probs

	def predict(self,x_tst,threshold=0.5,return_probs=False):
		probs=self.predict_proba(x_tst)
		y_pred=(probs>threshold)*1
		if return_probs:
			return y_pred,probs
		else:
			return y_pred

	def fit_predict_proba(self,x_mat,y_mat,x_tst):
		# for memory efficiency (don't want to change .fit signature)
		self.num_features=x_mat.shape[1]
		self.num_labels=y_mat.shape[1]
		self.num_points=x_mat.shape[0]
		self._initialize_projectors()
		probs=None
		for i in range(0,self.num_trees):
			print("Fitting Tree #",i)
			ctree = ClusterTree(**copy.deepcopy(self.static_tree_args))
			proj_x=self.projectors_x[i].project_data(x_mat)
			proj_y=self.projectors_y[i].project_data(y_mat)
			ctree.fit(proj_x,y_mat,proj_y)
			print("Predicting on Tree #",i)
			proj_x_tst=self.projectors_x[i].project_data(x_tst)
			probs_t=ctree.predict_proba(proj_x_tst)
			if probs is None:
				probs=probs_t
			else:
				probs+=probs_t
		probs=probs/float(self.num_trees)
		return probs

