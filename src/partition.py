import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
from networkx.algorithms.community.kernighan_lin import kernighan_lin_bisection
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize as normalize_features
import utils


class PartitioningAlgorithm:

	def __init__(self):
		pass

	def partition(self,x_mat):
		pass


class RandomPartitioner(PartitioningAlgorithm):

	def __init__(self,num_partitions=2,partition_ratios=None):
		assert(num_partitions>1)
		self.num_partitions=num_partitions
		if not partition_ratios: 
			# ignore partition_ratios if passed and create balanced partition
			partition_ratios=np.zeros(num_partitions)+1
		else:
			assert(len(partition_ratios)==num_partitions)
			partition_ratios=partition_ratios
		cumulative_ratios=np.cumsum(partition_ratios)
		cumulative_ratios=cumulative_ratios/cumulative_ratios[-1]
		assert(cumulative_ratios[-1]==1)
		self.cumulative_ratios=cumulative_ratios

	def partition(self,x_mat):
		num_samples=x_mat.shape[0]
		num_partitions=len(self.cumulative_ratios)
		# base case
		if num_partitions >= num_samples:
			padding = [[] for i in range(0,num_partitions-num_samples)]
			one_per_part=[[idx] for idx in range(0,num_samples)]
			return one_per_part + padding
		# generate a random permutation of the points
		permute=np.random.permutation(num_samples)
		# calculate the indices of each partition from cumulative_ratios
		partition_idcs=np.ceil(self.cumulative_ratios*num_samples).astype(dtype="int")
		# add the zero index
		partition_idcs=np.concatenate(([0],partition_idcs))
		assert(partition_idcs.shape[0]==num_partitions+1)
		partitions=[]
		for i in range(0,num_partitions):
			lpart=permute[partition_idcs[i]:partition_idcs[i+1]]
			partitions.append(lpart)
		return partitions


class KMeansPartitioner(PartitioningAlgorithm):

	def __init__(self,num_partitions=2,**kwargs):
		assert(num_partitions>1)
		self.num_partitions=num_partitions
		self.kmeanskwargs=kwargs

	def partition(self,x_mat):
		num_samples=x_mat.shape[0]
		assert(len(x_mat.shape)==2)
		# base case
		if self.num_partitions >= num_samples:
			padding = [[] for i in range(0,self.num_partitions-num_samples)]
			one_per_part=[[idx] for idx in range(0,num_samples)]
			return one_per_part + padding
		# run kmeans from scikit
		kmeans=KMeans(n_clusters=self.num_partitions, **self.kmeanskwargs)
		kmeans=kmeans.fit(x_mat)
		cluster_labels=kmeans.labels_
		# parse labels 
		partitions=[]
		for c in range(0,self.num_partitions):
			cluster_support=np.nonzero(cluster_labels==c)[0]
			partitions.append(cluster_support.tolist())
		return partitions


class BalancedKMeansPartitioner(PartitioningAlgorithm):
	
	def __init__(self,num_partitions=2,tol=0.003,max_iter=300):
		self.num_partitions=num_partitions
		self.tol=tol
		self.max_iter=max_iter

	def partition(self,x_mat):
		num_samples=x_mat.shape[0]
		assert(x_mat.shape[0]==num_samples)
		num_dims=x_mat.shape[1]
		# base case
		if self.num_partitions >= num_samples:
			padding = [[] for i in range(0,self.num_partitions-num_samples)]
			one_per_part=[[idx] for idx in range(0,num_samples)]
			return one_per_part + padding
		# maximum size of each balanced cluster
		max_size=np.ceil(num_samples/self.num_partitions)
		# initialize means
		permute=np.random.permutation(num_samples)
		means=x_mat[permute[:self.num_partitions],:]
		assert(means.shape[0]==self.num_partitions)
		# initialize loop vars
		new_means=np.zeros(means.shape)
		diff=np.inf
		num_iter=0
		sorted_clusters=[None]*self.num_partitions
		# loop to convergence / limit
		while diff > self.tol and num_iter < self.max_iter:
			# calculate distances between means and labels
			# numpy voodoo
			dist=utils.calculate_euclidean_distances(x_mat,means)
			assert(dist.shape==(num_samples,self.num_partitions))
			dist=np.sqrt(dist)
			
			# initialize sorted (w.r.t distance from mean)
			for i in range(0,self.num_partitions):
			 	sorted_clusters[i]=[]
			# assign new labels keeping clusters balanced
			for idx in range(0,num_samples):
				finished=False
				# insert idx into its best cluster, possibly setting off a cascade to maintain
				# size of cluster < max_size
				ins_idx=idx
				while not finished:
					best_cluster=np.argmin(dist[ins_idx,:])
					sorted_clusters[best_cluster]=self._insert_in_sorted_list(sorted_clusters[best_cluster],
												(ins_idx,dist[ins_idx,best_cluster]))
					# check if size is still ok
					if len(sorted_clusters[best_cluster])<=max_size:
						# done with the cascades (if any)
						finished=True
					else:
						# remove the worst/last label from this cluster
						ins_idx=sorted_clusters[best_cluster].pop(-1)[0]
						dist[ins_idx,best_cluster]=np.inf

			# calculate new means
			for i in range(0,self.num_partitions):
				idcs = [el[0] for el in sorted_clusters[i]]
				# print(x_mat[idcs,:].shape)
				new_means[i,:]=np.mean(x_mat[idcs,:],axis=0)
			# update
			diff=np.linalg.norm(means-new_means,ord=2)
			means=new_means
			num_iter=num_iter+1
		partitions=[None]*self.num_partitions
		for i in range(0,len(sorted_clusters)):
			partitions[i]=[j for (j,dist) in sorted_clusters[i]]
		return partitions

	def _insert_in_sorted_list(self,sorted_list,element):
		# insert element into a sorted list of (sample_id,sorting_key) pairs
		_,sorting_key=element
		# dummy element for easier indexing
		sorted_list=[(-50,-np.inf)]+sorted_list
		# linear search for correct position
		for i in range(0,len(sorted_list)):
			# reached the last position
			if i==len(sorted_list)-1:
				sorted_list.append(element)
				break
			if sorted_list[i][1]<=sorting_key and sorted_list[i+1][1]>sorting_key:
				sorted_list=sorted_list[:i+1]+[element]+sorted_list[i+1:]
				inserted=True
				break
		sorted_list=sorted_list[1:]
		return sorted_list


class ParabelBisector(PartitioningAlgorithm):

	def __init__(self,tol=0.004,max_iter=300):
		self.tol=tol
		self.max_iter=max_iter

	def _loss_value(self,alpha,products):
		assert(products.shape[1]==alpha.shape[0])
		assert(products.shape[0]==2)
		loss=np.sum(products[1,:] * (1/2+alpha/2))
		loss+=np.sum(products[0,:] * (1/2-alpha/2))
		loss=loss/alpha.shape[0]
		return loss

	def partition(self,x_mat):
		num_samples=x_mat.shape[0]
		num_dims=x_mat.shape[1]
		# normalize points
		x_mat=normalize_features(x_mat)
		# initialize 2 means randomly from points
		permute=np.random.permutation(num_samples)
		# 0 = -ve, 1 = +ve
		means=x_mat[permute[:2],:]
		# initialize other vars alpha
		alpha=np.zeros(num_samples)
		old_loss=np.inf
		loss=0
		i=0
		diff=np.inf
		# loop till convergence/max_iter
		while diff > self.tol and i < self.max_iter:
			# part a : minimize loss w.r.t alpha
			# calculate products/similarities (mu_+- mu_-)v
			products=utils.calculate_dot_products(means,x_mat)
			diff_prods=products[1,:]-products[0,:]
			# calculate rank of the prods - argsort(argsort(A))=rank(A)
			order=np.argsort(diff_prods)
			ranks=np.argsort(order)
			# need ranks 1-indexed
			signs_from=(ranks+1)-(num_samples+1)/2
			# get alpha from ranks
			alpha[signs_from>0]=1
			alpha[signs_from<0]=-1
			corner_case=np.nonzero(signs_from==0)[0]
			assert(len(corner_case)<=1)
			if len(corner_case)==1:
				mid=corner_case[0]
				alpha[mid]= 1 if diff_prods[mid] > 0 else -1
			# part b : minimize loss w.r.t means
			# means = sum (support) / norm
			means[1,:]=np.sum(x_mat[alpha==1,:],axis=0)
			means[0,:]=np.sum(x_mat[alpha==-1,:],axis=0)
			means=normalize_features(means)
			# calculate loss
			old_loss=loss
			loss=self._loss_value(alpha,products)
			i+=1
			# skip first iter
			if i>1:
				diff = np.abs(old_loss-loss)
		# get partitions
		partitions=[np.nonzero(alpha==1)[0].tolist(),
					np.nonzero(alpha==-1)[0].tolist()]
		return partitions


class KernighanLinGraphBisector(PartitioningAlgorithm):

	def __init__(self,num_all_labels,max_iter=300):
		self.max_iter=max_iter
		self.num_all_labels=num_all_labels

	def partition(self,x_mat):
		num_samples=x_mat.shape[0]
		assert(x_mat.shape[1]==self.num_all_labels+1)
		# the last dim of x_mat gives the id of the label
		# the other dims give that row of the coocc matrix
		label_ids=x_mat[:,-1]
		assert(np.all(label_ids<self.num_all_labels))
		adj_mat=x_mat[:,:-1][:,label_ids]
		# no self edges
		np.fill_diagonal(adj_mat,0)
		similarity_graph=nx.from_numpy_matrix(np.matrix(adj_mat))
		[p1,p2]=kernighan_lin_bisection(similarity_graph, partition=None, max_iter=self.max_iter, weight='weight')
		partitions=[p1,p2]
		return partitions


class SphericalKMeansPartitioner(PartitioningAlgorithm):

	class BadInitializationError(Exception):
	
		def __init__(self, msg):
			self.msg = msg


	def __init__(self,num_partitions=2,tol=0.003,max_iter=300,init="kmeans++",num_bad_tries=5):
		self.num_partitions=num_partitions
		self.tol=tol
		self.max_iter=max_iter
		self.init=init
		self.num_bad_tries=num_bad_tries

	def partition(self,x_mat):
		if type(x_mat)==csr_matrix:
			x_mat=x_mat.toarray()
		assert(np.all(np.isfinite(x_mat)))
		assert(np.any(np.isnan(x_mat))==False)
		assert(x_mat.shape[0]>0)
		# edge cases : some of the points are vec(0)
		# in this case cosine distance has no meaning and spherical k means will
		# give strange results
		num_nonzero_cols=np.sum((x_mat!=0)*1,axis=1)
		zero_idcs=np.nonzero(num_nonzero_cols==0)[0]
		nonzero_idcs=np.nonzero(num_nonzero_cols>0)[0]
		num_means=self.num_partitions
		if len(zero_idcs)==x_mat.shape[0]:
			# only zero points here
			return [zero_idcs.tolist()]
		elif len(zero_idcs)>0 and self.num_partitions==2:
			# just 2 partitions, divide as zero and nonzero points
			return [zero_idcs.tolist(),nonzero_idcs.tolist()]
		elif len(zero_idcs)>0 and self.num_partitions>2:
			assert(len(nonzero_idcs)>0)
			# reduce the number of means by 1
			num_means-=1
			# change the array_to_partition to only nonzero points
			x_mat=x_mat[nonzero_idcs,:]
		else:
			# nothing! len(zero_idcs)==0 , so all good to go
			assert(len(zero_idcs)==0)
	
		# run spherical k means at most num_bad_tries times
		partitions=None
		num_tries=0
		while partitions is None:
			try :
				partitions=self._spherical_k_means(x_mat,num_means)
			except BadInitializationError as bi:
				print("Oops! Bad means were encountered, restarting.")
				print(x_mat.shape)
			num_tries+=1
			if num_tries==self.num_bad_tries:
				assert False, "Too many tries"
		
		if len(zero_idcs)==0:
			# no problem, just return the partitions
			return partitions
		else:
			# partitions are w.r.t. nonzero_idcs
			# get indexes w.r.t original array
			new_partitions=[]
			for part in partitions:
				new_partitions.append(nonzero_idcs[part])
			# add the zero points
			partitions.append(zero_idcs.tolist())
			return partitions

	def _spherical_k_means(self,x_mat,num_means):
		# print(x_mat.shape)
		num_points=x_mat.shape[0]
		num_features=x_mat.shape[1]
		# normalize features
		x_mat=normalize_features(x_mat,norm="l2")
		# edge case : if number of unique points in x_mat_norm is < num_means
		unique_x_mat,unq_labels=np.unique(x_mat,axis=0,return_inverse=True)
		if unique_x_mat.shape[0]<=num_means:
			partitions=[]
			# just partition by unique value
			for i in range (0,unique_x_mat.shape[0]):
				idcs=np.nonzero(unq_labels==i)[0]
				partitions.append(idcs.tolist())
			return partitions
		# initialize means from points
		means=self._get_initial_means(x_mat,num_means)
		# initialize labels
		assignments=np.zeros(num_points)
		done=False
		num_iter=0
		# loop till convergence
		while not done and num_iter <= self.max_iter:
			# update loop vars
			num_iter+=1
			old_means=np.copy(means)
			# get dot products points x means
			dot_prods=utils.calculate_dot_products(x_mat,means)
			assert(dot_prods.shape==(num_points,num_means))
			# assign point to mean with largest dot prod/similarity
			assignments=np.argmax(dot_prods,axis=1)
			# calculate new means
			for m in range(0,num_means):
				support_bool=assignments==m
				if not np.any(support_bool):
					raise BadInitializationError("An empty partition was encountered!")
				means[m,:]=np.mean(x_mat[support_bool,:],axis=0)
			# normalize means
			means=normalize_features(means)
			# calculate dot prods with old means
			mdots=np.sum(means * old_means,axis=1)
			# if minimum distance is > 1 - e (at convergence dot=1)
			if np.min(mdots) >= 1 - self.tol:
				done=True
		# prepare partitions
		partitions=[None for i in range(0,num_means)]
		for i in range(0,num_means):
			idcs=np.nonzero(assignments==i)[0]
			assert(len(idcs)>0)
			partitions[i]=idcs.tolist()
		return partitions

	def _get_initial_means(self,x_mat,num_means):
		num_points=x_mat.shape[0]
		num_features=x_mat.shape[1]
		if self.init=="random":
			# make sure to select unique means (else will have empty partitions)
			means=np.zeros((num_means,num_features))
			while np.unique(means,axis=0).shape[0]<num_means:
				init_idcs=np.random.permutation(num_points)[:num_means]
				means=x_mat[init_idcs,:]
			return means

		elif self.init=="kmeans++":
			means_list=[]
			all_idcs=np.arange(0,num_points)
			# choose first mean randomly
			means_list.append(np.random.choice(all_idcs))
			while len(means_list)<num_means:
				means=x_mat[means_list,:]
				# calculate all distances
				dist_all=1-utils.calculate_dot_products(x_mat,means)
				# calculate best distance
				dist_best=np.min(dist_all,axis=1)
				assert(dist_all.shape==(x_mat.shape[0],means.shape[0]))
				# sample next mean from distribution proportional to d^2
				probs=np.square(dist_best)
				probs=probs/np.sum(probs)
				new_mean_idc=np.random.choice(all_idcs,p=probs)
				means_list.append(new_mean_idc)
			# get means
			means=x_mat[means_list,:]
			return means

