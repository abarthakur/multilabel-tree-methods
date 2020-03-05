import numpy as np
from sklearn.preprocessing import normalize as normalize_features
import sklearn.metrics as skmet
import pandas as pd 


def generate_parabel_label_representations(x_mat,y_mat):
	assert(y_mat.shape[0]==x_mat.shape[0])
	x_dim=x_mat.shape[1]
	num_labels=y_mat.shape[1]
	repre=np.zeros((num_labels,x_dim))
	# for each label calculate the mean of its support
	for l in range(0,num_labels):
		supp_idcs=np.nonzero(y_mat[:,l]>0)[0]
		if len(supp_idcs)==0:
			print("Warning! Label ",l,"has no positive examples. Setting representation to 0.")
			continue
		label_support=x_mat[supp_idcs,:]
		mean_support_vec=np.mean(label_support,axis=0)
		repre[l,:]=mean_support_vec
	# normalize features after
	repre=normalize_features(repre)
	return repre


def generate_cooccurrence_matrix(y_mat):
	coocc_mat=np.zeros((y_mat.shape[1],y_mat.shape[1]))
	for l in range(0,y_mat.shape[1]):
		support=y_mat[y_mat[:,l]>0,:]
		intersections=np.sum(support,axis=0)
		coocc_mat[l,:]=intersections
	return coocc_mat


def precision_at_k(y_tst,probs_pred,k):
	assert(k>0 and int(k)==k)
	assert(y_tst.shape[0]==probs_pred.shape[0])
	top_k=np.argsort(probs_pred,axis=1)[:,-k:]
	total=0
	for s_idx in range(0,y_tst.shape[0]):
		best_labels=top_k[s_idx,:]
		total+=np.sum(y_tst[s_idx,best_labels])
	p_at_k=total/y_tst.shape[0]
	p_at_k=p_at_k/k
	return p_at_k


def calculate_euclidean_distances(points_a,points_b):
	# note values computed will be slightly different
	# from np.linalg.norm but they are np.close
	n1=points_a.shape[0]
	n2=points_b.shape[0]
	num_dims=points_a.shape[1]
	assert(points_b.shape[1]==num_dims)
	# numpy broadcasting voodoo
	# Refer notes regarding test for this
	temp=points_a.reshape((n1,1,num_dims))
	temp=temp - points_b.reshape((1,n2,num_dims))
	assert(temp.shape==(n1,n2,num_dims))
	dist=np.sum(np.square(temp),axis=2)
	assert(dist.shape==(n1,n2))
	dist=np.sqrt(dist)
	return dist


def calculate_dot_products(points_a,points_b):
	n1=points_a.shape[0]
	n2=points_b.shape[0]
	num_dims=points_a.shape[1]
	assert(points_b.shape[1]==num_dims)
	temp=points_a.reshape(n1,1,num_dims)
	temp=temp * points_b.reshape(1,n2,num_dims)
	dot_prods=np.sum(temp,axis=2)
	assert(dot_prods.shape==(n1,n2))
	return dot_prods


def calculate_performance_metrics(y_tst,y_pred,probs_pred):
	metrics_df=pd.DataFrame(index=[0])
	if y_pred is not None:
		metrics_df.loc[0,"prec_micro"]=skmet.precision_score(y_tst,y_pred,average='micro')
		metrics_df.loc[0,"prec_macro"]=skmet.precision_score(y_tst,y_pred,average='macro')
		metrics_df.loc[0,"rec_micro"]=skmet.recall_score(y_tst,y_pred,average='micro')
		metrics_df.loc[0,"rec_macro"]=skmet.recall_score(y_tst,y_pred,average='macro')
		metrics_df.loc[0,"f1_micro"]=skmet.f1_score(y_tst,y_pred,average='micro')
		metrics_df.loc[0,"f1_macro"]=skmet.f1_score(y_tst,y_pred,average='macro')
		metrics_df.loc[0,"hamming"]=skmet.hamming_loss(y_tst,y_pred)
	if probs_pred is not None:
		metrics_df.loc[0,"p@1"]=precision_at_k(y_tst,probs_pred,1)
		metrics_df.loc[0,"p@3"]=precision_at_k(y_tst,probs_pred,3)
		metrics_df.loc[0,"p@5"]=precision_at_k(y_tst,probs_pred,5)
		metrics_df.loc[0,"ranking_loss"]=skmet.label_ranking_loss(y_tst,probs_pred)
		metrics_df.loc[0,"coverage_error"]=skmet.coverage_error(y_tst,probs_pred)
		metrics_df.loc[0,"avg_prec_score"]=skmet.label_ranking_average_precision_score(y_tst,probs_pred)
		# calculate ndcg@k
		metrics_df.loc[0,"ndcg@1"]=skmet.ndcg_score(y_tst,probs_pred,1)
		metrics_df.loc[0,"ndcg@3"]=skmet.ndcg_score(y_tst,probs_pred,3)
		metrics_df.loc[0,"ndcg@5"]=skmet.ndcg_score(y_tst,probs_pred,5)
		# calculate dcg@k
		metrics_df.loc[0,"dcg@1"]=skmet.dcg_score(y_tst,probs_pred,1)
		metrics_df.loc[0,"dcg@3"]=skmet.dcg_score(y_tst,probs_pred,3)
		metrics_df.loc[0,"dcg@5"]=skmet.dcg_score(y_tst,probs_pred,5)
	return metrics_df