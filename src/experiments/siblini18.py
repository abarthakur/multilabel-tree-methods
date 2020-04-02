import sys
sys.path.append("..")

import os
import utils
import mydatasets
import os
import pandas as pd
import time
import copy
from craftml import CraftMLEnsemble


class Siblini18Experiment:

	def __init__(self,dataset_name,num_repititions,outfilename,splits=None):
		self.outfilename=outfilename
		self.num_repititions=num_repititions
		self.dataset_name=dataset_name
		self.splits=splits
		# static tree arguments
		if dataset_name in ["mediamill","bibtex","delicious"]:
			self.has_splits=True
		elif dataset_name in ["eurlex"]:
			self.has_splits=False


	def run_experiment(self):
		if self.has_splits:
			self.run_experiment_small_datasets()
		else:
			self.run_experiment_large_datasets()


	def run_experiment_large_datasets(self):
		curr_dir=os.getcwd()
		os.chdir("..")
		trn_data,tst_data=mydatasets.load_large_dataset(self.dataset_name)
		os.chdir(curr_dir)
		x_mat,y_mat=mydatasets.get_arrays(trn_data)
		x_tst,y_tst=mydatasets.get_arrays(tst_data)
		all_results=None
		# loop over reps, trees
		for rep_num in range(0,self.num_repititions):
			df=self.run_basic_experiment(x_mat,y_mat,x_tst,y_tst)
			df.loc[0,"dataset"]=self.dataset_name
			df.loc[0,"split"]=0
			df.loc[0,"rep_num"]=rep_num
			if all_results is None:
				all_results=df
			else:
				all_results=all_results.append(df,ignore_index=True)
			with open(self.outfilename,"w") as fo:
				all_results.to_csv(fo,index=False)


	def run_experiment_small_datasets(self):
		# load dataset
		curr_dir=os.getcwd()
		os.chdir("..")
		dataset,trn_splits,tst_splits=mydatasets.load_dataset(self.dataset_name)
		os.chdir(curr_dir)
		all_results=None
		# loop over splits, reps, trees
		for split_num in self.splits:
			trn_data,tst_data=mydatasets.get_small_dataset_split(dataset,trn_splits,tst_splits,split_num)
			x_mat,y_mat=mydatasets.get_arrays(trn_data)
			x_tst,y_tst=mydatasets.get_arrays(tst_data)
			for rep_num in range(0,self.num_repititions):
				df=self.run_basic_experiment(x_mat,y_mat,x_tst,y_tst)
				df.loc[0,"dataset"]=self.dataset_name
				df.loc[0,"split"]=split_num
				df.loc[0,"rep_num"]=rep_num
				if all_results is None:
					all_results=df
				else:
					all_results=all_results.append(df,ignore_index=True)
				with open(self.outfilename,"w") as fo:
					all_results.to_csv(fo,index=False)


	def run_basic_experiment(self,x_mat,y_mat,x_tst,y_tst):
		ensemble = CraftMLEnsemble()
		probs_pred=ensemble.fit_predict_proba(x_mat,y_mat,x_tst)
		return utils.calculate_performance_metrics(y_tst,None,probs_pred)


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
	
	Siblini18Experiment(**args).run_experiment()