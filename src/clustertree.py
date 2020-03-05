import numpy as np
import copy

class ClusterTree:

	def __init__(self,partitioner,leaf_classifier,internal_classifier,stopping_condition):
		# tree building params
		self.partitioner=partitioner
		self.stopping_condition=stopping_condition
		# classification params
		self.leaf_classifier=leaf_classifier
		self.internal_classifier=internal_classifier
		# initialize tree attrs
		self.nodes=[]
		self.root=None

	def _get_new_node(self,parent,train_idcs,depth):
		new_node=ClusterTreeNode(self,parent,train_idcs,depth)
		new_node.idx=len(self.nodes)
		self.nodes.append(new_node)
		return new_node

	def fit(self,x_mat,y_mat,repre):
		assert(x_mat.shape[0]==y_mat.shape[0])
		assert(x_mat.shape[0]==repre.shape[0])
		self.num_features=x_mat.shape[1]
		self.num_labels=y_mat.shape[1]
		self.num_points=x_mat.shape[0]
		self._fit_tree(x_mat,y_mat,repre)
		self._fit_classifiers(x_mat,y_mat)

	def _fit_tree(self,x_mat,y_mat,repre):
		all_idcs=list(range(0,x_mat.shape[0]))
		self.root=self._get_new_node(None,all_idcs,0)
		self.root._split(x_mat,y_mat,repre)
		self.num_points=x_mat.shape[0]

	def _fit_classifiers(self,x_mat,y_mat):
		print("Fitting ",len(self.nodes)," classifiers...")
		for i,node in enumerate(self.nodes):
			node._fit(x_mat,y_mat)
		print("Done fitting")

	def predict_proba(self,x_tst):
		assert(x_tst.shape[1]==self.num_features)
		probs=np.zeros((x_tst.shape[0],self.num_labels))
		tst_idcs=list(range(0,x_tst.shape[0]))
		self.root._predict_proba(x_tst,probs,tst_idcs)
		return probs

	def predict(self,x_tst,threshold=0.5,return_probs=False):
		probs=self.predict_proba(x_tst)
		y_pred=(probs>threshold)*1
		if return_probs:
			return y_pred,probs
		else:
			return y_pred

	def walk_tree(self,walker):
		self.root._walk(walker)


class ClusterTreeNode:

	def __init__(self,ctree,parent,train_idcs,depth):
		self.tree=ctree
		self.train_idcs=train_idcs
		self.depth=depth
		self.parent=parent
		self.children=[]
		self.node_type=None
		self.classifier=None

	def _split(self,x_mat,y_mat,repre):
		# check if we have reached a leaf
		if self.tree.stopping_condition.check(self,x_mat,y_mat,repre):
			# print("Leaf with",len(self.train_idcs))
			self.node_type="leaf"
			return
		# otherwise split this node
		self.node_type="internal"
		# partition data
		repre_node=repre[self.train_idcs,:]
		dparts=self.tree.partitioner.partition(repre_node)
		dparts=[dpart for dpart in dparts  if dpart!=[] ]
		del repre_node
		# partitioning failed for some reason, and gave a trivial partition
		if len(dparts)<=1:
			self.node_type="leaf"
			return
		# for each partition instantiate a child node
		for i,dpart in enumerate(dparts):
			# translate indices to original array
			child_train_idcs=[self.train_idcs[idx] for idx in dpart]
			child=self.tree._get_new_node(self,child_train_idcs,self.depth+1)
			self.children.append(child)
			child._split(x_mat,y_mat,repre)

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
			y_mat_new=y_mat_new[self.train_idcs,:]
		elif self.node_type=="leaf":
			y_mat_new=y_mat[self.train_idcs,:]
		x_mat_new=x_mat[self.train_idcs,:]
		self.classifier.fit(x_mat_new,y_mat_new)

	def _predict_proba(self,x_tst_global,probs_global,tst_idcs):
		if self.node_type=="internal":
			# assume multiclass not multilabel classification
			routing_probs=self.classifier.predict_proba(x_tst_global[tst_idcs,:])
			routing_labels=np.argmax(routing_probs,axis=1)
			for ch_idx,child in enumerate(self.children):
				supp_idcs=np.nonzero(routing_labels==ch_idx)[0]
				if len(supp_idcs)==0:
					continue
				# translate to global idcs
				supp_idcs=[tst_idcs[idx] for idx in supp_idcs]
				child._predict_proba(x_tst_global,probs_global,supp_idcs)
		else:
			# get soft labels/probs from classifier
			probs=self.classifier.predict_proba(x_tst_global[tst_idcs])
			probs_global[tst_idcs,:]=probs
			return probs

	def _walk(self,walker):
		walker.process_node(self)
		for child in self.children:
			child._walk(walker)
