import numpy as np
import copy


class LabelTree:

	def __init__(self, partitioner, leaf_classifier, 
					internal_classifier, stopping_condition):
		# tree params
		self.partitioner=partitioner
		self.stopping_condition=stopping_condition
		# classification params
		self.leaf_classifier=leaf_classifier
		self.internal_classifier=internal_classifier
		# initialize tree params
		self.root=None
		self.nodes=[]

	def _get_new_node(self,parent,node_labels,train_idcs,depth):
		new_node=LabelTreeNode(self,parent,node_labels,train_idcs,depth)
		new_node.idx=len(self.nodes)
		self.nodes.append(new_node)
		return new_node

	def fit(self,x_mat,y_mat,repre):
		assert(x_mat.shape[0]==y_mat.shape[0])
		assert(repre.shape[0]==y_mat.shape[1])
		self.num_features=x_mat.shape[1]
		self.num_labels=y_mat.shape[1]
		self.num_trn_points=x_mat.shape[0]
		self._fit_tree(x_mat,y_mat,repre)
		self._fit_classifiers(x_mat,y_mat)

	def _fit_tree(self,x_mat,y_mat,repre):
		all_idcs=list(range(0,self.num_trn_points))
		all_labels=list(range(0,self.num_labels))
		self.root=self._get_new_node(None,all_labels,all_idcs,0)
		self.root._split(x_mat,y_mat,repre)

	def _fit_classifiers(self,x_mat,y_mat):
		print("Fitting ",len(self.nodes)," classifiers...")
		for i,node in enumerate(self.nodes):
			node._fit(x_mat,y_mat)
		print("Done fitting")

	def predict_proba(self,x_tst,method="beam_search",num_paths=10,recurse_threshold=0.5):
		assert(x_tst.shape[1]==self.num_features)
		assert(method in ["beam_search","recursive"])
		if method=="recursive":
			probs=np.zeros((x_tst.shape[0],self.num_labels))
			tst_idcs=list(range(0,x_tst.shape[0]))
			self.root._predict_proba(x_tst,probs,tst_idcs,recurse_threshold)
			return probs
		elif method=="beam_search":
			return self._predict_proba_beam_search(x_tst,num_paths)

	def predict(self,x_tst,threshold=0.5,method="beam_search",num_paths=10,
				recurse_threshold=0.5,return_probs=False):
		probs=self.predict_proba(x_tst,method,num_paths,recurse_threshold)
		y_pred=(probs>threshold)*1
		if return_probs:
			return y_pred,probs
		else:
			return y_pred

	def _predict_proba_beam_search(self,x_tst,num_paths):
		# vectorized beam search! way faster than calling per sample
		# but bug : in case of ties
		num_nodes=len(self.nodes)
		num_samples=x_tst.shape[0]
		# for each sample store the boundary as an array of path probs (to a node)
		boundary=np.zeros((num_samples,num_nodes))
		search_done=np.zeros(num_samples).astype(bool)
		# initialize loop variables
		boundary[:,self.root.idx]=1.0
		num_iter=0
		while not np.all(search_done):
			# get nodes which have at least one sample at it, and set search_done to true
			active_nodes_list=np.nonzero(np.sum(boundary,axis=0)>0)[0]
			# expand internal nodes in active_nodes_list
			search_done[:]=True
			for node_idx in active_nodes_list:
				node=self.nodes[node_idx]
				# active nodes which are leaves are skipped 
				# ~ thus for points at leaves, search_done=True
				if node.node_type=="leaf":
					continue
				# otherwise expand this node and mark search_done=False for its points
				children_global_idcs=[self.nodes.index(ch) for ch in node.children]
				active_points=np.nonzero(boundary[:,node_idx])[0]
				search_done[active_points]=False
				# route points to children by assigning values to 
				routing_probs=node.classifier.predict_proba(
									x_tst[active_points,:].reshape((-1,self.num_features)))
				assert(routing_probs.shape==(len(active_points),len(children_global_idcs)))
				path_probs= (routing_probs.T * boundary[active_points,node_idx] ).T
				for ch_par_idx,ch_glo_idx in enumerate(children_global_idcs):
					boundary[active_points,ch_glo_idx]=path_probs[:,ch_par_idx]
				# erase parent from boundary
				boundary[active_points,node_idx]=0
			# filter out all except best num_paths nodes for each sample
			best_nodes_per_sample=np.argsort(boundary,axis=1)[:,-num_paths:]
			new_boundary=boundary.copy()*0
			for s_idx in range(0,num_samples):
				for best_idx in range(0,num_paths):
					best_node_idx=best_nodes_per_sample[s_idx,-(best_idx+1)]
					# boundary is < num_paths
					if boundary[s_idx,best_node_idx]==0:
						break
					new_boundary[s_idx,best_node_idx]=boundary[s_idx,best_node_idx]
			boundary=new_boundary
			num_iter+=1
	
		# the active_nodes_list should be just leaves now, so lets fill the final probability arr
		# by multiplying path probs (boundary) with leaf probs
		active_nodes_list=np.nonzero(np.sum(boundary,axis=0)>0)[0]
		probs=np.zeros((num_samples,self.num_labels))
		for node_idx in active_nodes_list:
			node=self.nodes[node_idx]
			assert(node.node_type=="leaf")
			active_points=np.nonzero(boundary[:,node_idx])[0]
			probs_leaf=node.classifier.predict_proba(x_tst[active_points,:].reshape((-1,self.num_features)))
			total_probs=(probs_leaf.T * boundary[active_points,node_idx] ).T
			# assign probs
			for i,lab in enumerate(node.labels):
				probs[active_points,lab]=total_probs[:,i]
		return probs

	def walk_tree(self,walker):
		self.root._walk(walker)


class LabelTreeNode:

	def __init__(self,tree,parent,node_labels,train_idcs,depth):
		# labels values refer to original labelset (0,y_mat.shape[1]-1)
		self.labels=list(node_labels)
		# x_idcs refer to the training set (x_mat,y_mat)
		self.train_idcs=list(train_idcs)
		self.tree=tree
		self.depth=depth
		self.parent=parent
		self.node_type=None
		self.children=[]
		self.classifier=None

	def _check_partitions(self,lparts,dparts):
		union_labels=set()
		union_data=set()
		for i in range(0,len(dparts)):
			union_labels=union_labels|set(lparts[i])
			union_data=union_data|set(dparts[i])
		assert(union_labels==set(self.labels))
		if self.parent is not None:
			assert(union_data==set(self.train_idcs))

	def _split(self,x_mat,y_mat,repre):
		# check if we have reached a leaf
		if self.tree.stopping_condition.check(self,x_mat,y_mat,repre):
			self.node_type="leaf"
			return
		self.node_type="internal"
		# partition labels and filter empty partiyions
		lparts_temp=self.tree.partitioner.partition(repre[self.labels,:])
		lparts=[]
		for lpart in lparts_temp:
			if lpart==[]:
				continue
			# translate to original indices
			lp=[self.labels[lab] for lab in lpart]
			lparts.append(lp)
		# partition data
		dparts=self._partition_data_by_labels(lparts,y_mat)
		self._check_partitions(lparts,dparts)
		if len(dparts)<=1:
			self.node_type="leaf"
			return
		# for each partition instantiate a new node
		for i in range(0,len(lparts)):
			child=self.tree._get_new_node(self,lparts[i],dparts[i],self.depth+1)
			self.children.append(child)
			child._split(x_mat,y_mat,repre)

	def _partition_data_by_labels(self,label_partitions,y_mat):
		# check that all labels are valid indices
		for lpart in label_partitions:
			for label in lpart:
				assert(label<y_mat.shape[1])
		# partition data
		data_partitions=[]
		for lpart in label_partitions:
			if lpart==[]:
				continue
			# get points where at least one of the labels is active / =1
			active_bool=np.sum(y_mat[:,lpart],axis=1) > 0
			data_idcs=np.nonzero(active_bool)[0]
			data_partitions.append(data_idcs.tolist())
		return data_partitions

	def _fit(self,x_mat,y_mat):
		if self.node_type=="leaf":
			self.classifier=copy.deepcopy(self.tree.leaf_classifier)
		else:
			self.classifier=copy.deepcopy(self.tree.internal_classifier)
		y_mat_new=None
		if self.node_type=="internal":
			# for internal nodes, generate pseudo labels for children
			y_mat_new=np.zeros((y_mat.shape[0],len(self.children)))
			for i,child in enumerate(self.children):
				y_mat_new[child.train_idcs,i]=1
			# restrict to active points only
			y_mat_new=y_mat_new[self.train_idcs,:]
		elif self.node_type=="leaf":
			# restrict to active points only with active labels
			y_mat_new=y_mat[self.train_idcs,:][:,self.labels]
		# restrict x_mat to active points only
		x_mat_new=x_mat[self.train_idcs,:]
		self.classifier.fit(x_mat_new,y_mat_new)

	def _predict_proba(self,x_tst_global,probs_global,tst_idcs,recurse_threshold):
		if self.node_type=="internal":
			routing_probs=self.classifier.predict_proba(x_tst_global[tst_idcs,:])
			routing_labels=(routing_probs>recurse_threshold)*1
			for ch_idx,child in enumerate(self.children):
				child_tst_idcs=np.nonzero(routing_labels[:,ch_idx]>0)[0]
				# translate to original array
				child_tst_idcs=[tst_idcs[idx] for idx in child_tst_idcs]
				if len(child_tst_idcs)>0:
					child._predict_proba(x_tst_global,probs_global,child_tst_idcs,recurse_threshold)
		if self.node_type=="leaf":
			probs_leaf=self.classifier.predict_proba(x_tst_global[tst_idcs,:])
			for i,lab in enumerate(self.labels):
				probs_global[tst_idcs,lab]=probs_leaf[:,i]

	def _walk(self,walker):
		walker.process_node(self)
		for child in self.children:
			child._walk(walker)


class LeafSizeStoppingCondition:

	def __init__(self,min_leaf_size):
		assert(min_leaf_size>0)
		self.min_leaf_size=min_leaf_size
	
	def check(self,lnode,x_mat,y_mat,repre):
		return len(lnode.labels) <= self.min_leaf_size
