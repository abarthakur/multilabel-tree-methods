import numpy as np
import graphviz

class Walker:
	
	def __init__(self):
		pass
	
	def process_node(self,node):
		pass


class GraphvizBuilder(Walker):
	
	def __init__(self,fmt="png"):
		self.graph=graphviz.Digraph(format=fmt)
	
	def process_node(self,lnode):
		self.graph.node(str(lnode.idx))
		if lnode.parent:
			self.graph.edge(str(lnode.parent.idx),str(lnode.idx))

class DataCounter(GraphvizBuilder):
	
	def __init__(self):
		super().__init__()
	
	def process_node(self,lnode):
		super().process_node(lnode)
		graph=self.graph
		graph.node(str(lnode.idx),str(len(lnode.train_idcs)))


class LabelCounter(GraphvizBuilder):
	
	def __init__(self):
		super().__init__()
	
	def process_node(self,lnode):
		super().process_node(lnode)
		graph=self.graph
		graph.node(str(lnode.idx),str(len(lnode.labels)))


class DataImbalanceWalker(GraphvizBuilder):
	
	def __init__(self,y_mat,print_idx=True):
		self.print_idx=print_idx
		self.y_mat=y_mat
		super().__init__()
	
	def process_node(self,lnode):
		super().process_node(lnode)
		graph=self.graph
		parent_tcount=len(lnode.train_idcs)
		children_tcounts=[]
		if lnode.node_type=="internal":
			union_children=set()
			for c in lnode.children:
				children_tcounts.append(len(c.train_idcs))
				union_children=union_children|set(c.train_idcs)
		elif lnode.node_type=="leaf":
			children_tcounts=list(np.sum(self.y_mat[:,lnode.labels],axis=0))
		children_ratios=[tc/(parent_tcount-tc) if tc!=parent_tcount else -1 for tc in children_tcounts ]
		node_label_string=",".join([str(round(r,3)) for r in children_ratios])
		if self.print_idx:
			node_label_string=str(lnode.idx)+" || "+node_label_string
		graph.node(str(lnode.idx),node_label_string)