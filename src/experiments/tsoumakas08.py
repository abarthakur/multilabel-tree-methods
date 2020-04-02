## Imports
import sys
sys.path.append("..")

import os
import pandas as pd
import time
import mydatasets
import utils
from labeltree import LabelTree,LeafSizeStoppingCondition
from partition import RandomPartitioner,KMeansPartitioner,BalancedKMeansPartitioner
from classifiers import OVAMultiLabelClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB

class Tsoumakas2008Experiment:

	def __init__(self,dataset_name,splits,num_repititions,outfilename):
		assert(dataset_name in ["mediamill","delicious"])
		assert(num_repititions>0)
		self.outfilename=outfilename
		self.num_repititions=num_repititions
		self.dataset_name=dataset_name
		self.splits=splits
		# load data
		curr_dir=os.getcwd()
		os.chdir("..")
		self.dataset, self.trn_splits, self.tst_splits = mydatasets.load_dataset(self.dataset_name)
		os.chdir(curr_dir)
		# static label tree arguments
		base= GaussianNB() if dataset_name=="mediamill" else BernoulliNB()
		self.static_args={
							'stopping_condition' : LeafSizeStoppingCondition(1),
							'leaf_classifier' : OVAMultiLabelClassifier(base),
							'internal_classifier' : OVAMultiLabelClassifier(base)
						 }
		#partitioning methods
		self.methods=[RandomPartitioner,KMeansPartitioner,BalancedKMeansPartitioner]
	

	def run_experiment(self):
		all_results=None
		for split_num in self.splits:
			trn_data,tst_data=mydatasets.get_small_dataset_split(self.dataset,self.trn_splits,self.tst_splits,split_num)
			x_trn,y_trn=mydatasets.get_arrays(trn_data)
			x_tst,y_tst=mydatasets.get_arrays(tst_data)
			# columns of y_mat are label representation in Tsoumakas08
			repre=y_trn.T
			for method in self.methods:
				for num_partitions in range(2,9):
					for rep_num in range(0,self.num_repititions):
						partitioner=method(num_partitions)
						row_name=str(method.__name__)+"_s="+str(split_num)+"_k="+str(num_partitions)+"/r="+str(rep_num)
						print(row_name)
						df=self.train_and_test(repre,x_trn,y_trn,x_tst,y_tst,partitioner)
						df.loc[0,"row_name"]=row_name
						df.loc[0,"partitioner"]=str(method.__name__)
						df.loc[0,"split"]=split_num
						df.loc[0,"k"]=num_partitions
						df.loc[0,"r"]=rep_num
						if all_results is None:
							all_results=df
						else:
							all_results=all_results.append(df,ignore_index=True)
						with open(self.outfilename,"w") as fo:
							all_results.to_csv(fo,index=False)


	def train_and_test(self,repre,x_trn,y_trn,x_tst,y_tst,partitioner):
		# init
		ltree=LabelTree(partitioner=partitioner, **self.static_args)
		ltree.num_features=x_trn.shape[1]
		ltree.num_labels=y_trn.shape[1]
		ltree.num_trn_points=x_trn.shape[0]
		# build tree
		part_time=time.perf_counter()
		ltree._fit_tree(x_trn,y_trn,repre)
		part_time=time.perf_counter()-part_time
		# train classifiers
		trn_time=time.perf_counter()
		ltree._fit_classifiers(x_trn,y_trn)
		trn_time=time.perf_counter()-trn_time
		# predict on test data
		tst_time=time.perf_counter()
		y_pred,probs_pred=ltree.predict(x_tst,threshold=0.5,method="recursive",
							num_paths=None,recurse_threshold=0.5,return_probs=True)
		tst_time=time.perf_counter()-tst_time
		# get performance metrics
		df=utils.calculate_performance_metrics(y_tst,y_pred,probs_pred)
		# add time metrics
		df.loc[0,"partition_time"]=part_time
		df.loc[0,"train_time"]=trn_time
		df.loc[0,"test_time"]=tst_time
		return df


if __name__=="__main__":
	args={}
	for i in range(1,len(sys.argv)):
		string=sys.argv[i]
		parts=string.split("=")
		assert(len(parts)==2)
		if parts[0]=="dataset":
			args["dataset_name"]=parts[1]
		if parts[0]=="rep":
			# oops, spelling error
			args["num_repititions"]=int(parts[1])
		if parts[0]=="splits":
			args["splits"]=[int(k) for k in parts[1].split(",")]
		if parts[0]=="out":
			with open(parts[1],"a") as fo:
				pass
			args["outfilename"]=parts[1]
	
	Tsoumakas2008Experiment(**args).run_experiment()