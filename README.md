### Project Overview

As part of my Bachelor's Thesis Project (BTP), I worked on developing a method for extreme multilabel classification

* The method is a variant of a Label Tree, which consist of recursively partitioning the labels into a tree structure with leaves containing classifiers that predict the relevance of one or more labels, and internal nodes containing classifiers that predict the relevance of its child subtree. Label Tree methods offer logarithmic time complexity in evaluation, compared to the baseline ensemble method of Binary Relevance (One-vs-all).
* Particularly, the idea was to improve the partitioning or clustering step, by representing the labels as a graph of cooccurrences and utilizing graph clustering/community detection algorithms to find meaningful partitions.
* Examples of Label Trees include [Parabel][2] and [HOMER][1]

When I returned to the project later, I also looked at Cluster Tree methods, specifically CraftML, to possibly leverage locality of labelsets in the feature space. After expending some more time, I ultimately abandoned this direction (loosely, tree ensemble approaches). 

I have gathered all useful code from my efforts into this repository, which may serve as a useful starting point.

> This repository is a python implementation of [Parabel][2] and [HOMER][1] (Label tree methods, and [CraftML][3] (Cluster tree method) written in a modular way so that you can switch out various components. 

> It also contains notebooks with exploratory data analysis of multi-label datasets, and analysis of the tree methods themselves.


### Table of Contents

* [Installation](#installation)
* [Notebooks](#notebooks)
* [Modules](#modules)
	* [labeltree.py](#src/labeltree.py)
	* [clustertree.py](#src/clustertree.py)
	* [partition.py](#src/partition.py)
	* [classifiers.py](#src/classifiers.py)
	* [nn_classifiers.py](#src/nn_classifiers.py)
	* [craftml.py](#src/craftml.py)
	* [utils.py](#src/utils.py)
* [Resources and References](#resources-and-references)

### Installation

To run, first install the required Python 3 modules as written in ``requirements.txt``. All library versions are set to the ones I used during development. In a fresh environment, ``pip install -r requirements.txt`` should do the job!

### Notebooks

* [src/notebooks/eda_labels.ipynb](/src/notebooks/eda_labels.ipynb) : Contains various graphs and statistics to investigate how label counts vary, and how much overlap there is between labels.
* [src/notebooks/eda_tsne_visualization.ipynb](/src/notebooks/eda_tsne_visualization.ipynb): Contains interactive visualizations of labels in feature space, labels in labelset space and labelsets in feature space. Purpose is to determine if labels/labelsets are localized in X space, or if labels are localized in Y space.
* [src/notebooks/labeltree_imbalance_visualization.ipynb](/src/notebooks/labeltree_imbalance_visualization.ipynb) : Visualizes data imbalance in the intermediate binary classification problems at nodes and leaves of label trees, with different choices for the clustering.

### Modules

#### src/labeltree.py

**LabelTree**

Label trees methods follow the following general framework

1. Choose some representation for the labels. For ex: [HOMER][1] chooses the columns of the label matrix. 
2. Recursively cluster the labels into a hierarchy. Leaves may contain one or more labels. The labelsets of an internal node is the union of its leaves.
3. At internal nodes, learn classifiers for each child predicting the probability that the labelset of the child is relevant to the sample.
3. At leaf nodes, learn classifiers modelling the probability of each label, given that the leaf's labelset is relevant to the sample.

Given a novel sample, there are two choices implemented for evaluation
* **recursive**: This is the strategy utilized by [HOMER][1], where you recursively predict one or more paths down to leaves, depending on whether at an internal node, a child node's probability is greater than ``recurse_threshold``. Thereafter the probability of a label is 0 if it was not reached, otherwise the value predicted at the leaf.
*  **beam_search**: This is the strategy presented in [Parabel][2] which is more theoretically grounded. It relies on the interpretation that the label tree's nodes all model conditional probabilities. Then the marginal probability of a label is the product of the conditional probabilities along its path from the root. Thus, in the interest of efficiency, only a few of the paths will be evaluated, which can be chosen by any graph search strategy like best-first or beam search. In this case beam search has been implemented.

*Parameters* :

* partitioner : Instance of PartitioningAlgorithm used to recursively cluster the labels.
* leaf_classifier : Classifier to use in leaf node. This should be a multilabel classifier.
* internal_classifier : Classifier to use in internal node. This should be a multilabel classifier.
* stopping_condition : An object implementing the method ``check(lnode,x_mat,y_mat,repre)`` used to check whether to stop clustering. An example is ``LeafSizeStoppingCondition`` in ``src/labeltree.py``.

> fit((x_mat,y_mat,repre)

Generates the cluster tree and fits the classifiers at its nodes for their respective subproblems.
* x_mat : Numpy array of shape (num_samples,num_features) representing features to use for classification at internal and leaf nodes.
* y_mat : Numpy array of shape (num_samples,num_labels) representing label vectors to use for classification at internal and leaf nodes.
* repre : Numpy array of shape (num_labels,repre_dim). This can be any custom representation of the labels, for example, representation for Parabel can be generated using ``generate_parabel_label_representations``.

> predict_proba(x_tst,method="beam_search",num_paths=10,recurse_threshold=0.5)

Predict soft labels/probabilities for samples in x_tst.

* x_tst : Numpy array of shape (N,num_features).
* method : "recursive" or "beam_search".
* num_paths : number of paths for ``beam_search`` strategy.
* recurse_threshold : threshold for exploring a node in ``recursive`` strategy.

> predict(self,x_tst,threshold=0.5,method="beam_search",num_paths=10,
				recurse_threshold=0.5,return_probs=False)

Predict hard labels for samples in x_tst, thresholding results from ``predict_proba``.


#### src/clustertree.py

**ClusterTree**

Cluster trees/Instance trees methods follow the following general framework

1. Recursively cluster training data points into a hierarchy. The representation of the points may be the features, or the labels.
2. At internal nodes, learn classifiers for each child predicting the probability that a novel sample belongs to that cluster given that it belongs to the cluster of the parent.
3. At leaf nodes, model the probability of **all** labels, given that the novel sample belongs to the leaf's cluster.
4. Given a novel sample, recursively predict a path from the root to a *single* leaf. Return the output of the classiier at the leaf as the predicted (soft) labels.

*Parameters* : 

* partitioner : Instance of PartitioningAlgorithm used to recursively cluster the data.
* leaf_classifier : Classifier to use in leaf node. This should be a multilabel classifier.
* internal_classifier : Classifier to use in internal node. In this implementation, this is assumed to be a multiclass classifier.
* stopping_condition : An object implementing the method ``check(cnode,x_mat,y_mat,repre)`` used to check whether to stop clustering. An example is ``CraftMLStoppingCondition`` in ``src/craftml.py``.

*Methods* :

> fit((x_mat,y_mat,repre)

Generates the cluster tree and fits the classifiers at its nodes for their respective subproblems.

* x_mat : Numpy array/ Or scikit sparse matrix of shape (num_samples,num_features) representing features to use for classification at internal and leaf nodes.
* y_mat : Numpy array/ Or scikit sparse matrix of shape (num_samples,num_labels) representing label vectors to use for classification at internal and leaf nodes.
* repre : Numpy array/ Or scikit sparse matrix of shape (num_samples,repre_dim). This can be any custom representation of the data points, for example, CraftML uses a hashing projection of y_mat for repre. ``repre`` is used only for building the hierarchy, and not for classification (subproblems).

> predict_proba(x_tst)

Predict soft labels/probabilities for samples in x_tst (for all labels).

> predict(x_tst,threshold=0.5,return_probs=False)

Predict hard labels for samples in x_tst, thresholding results from ``predict_proba``.

#### src/partition.py 

This file contains a collection of clustering algorithms for use by ``LabelTree`` and ``ClusterTree`` classes.

* **PartitioningAlgorithm** : Base class/Interface.
* **RandomPartitioner** : Choose from the space of partitions where each partition is of equal size (or sizes in a specified ratio) uniformly at random.
* **KMeansPartitioner** : Partition the data points using ``scikit-learn``'s implementation of K-Means (with stadndard L2 distance).
* **BalancedKMeansPartitioner** : Partition the data points into k balanced (in terms of size) clusters according to the algorithm described in [Tsoumakas08]
* **ParabelBisector** : Partition the data into 2 perfectly balanced clusters, minimizing the cosine distance from the mean, as described in [Prabhu18].
* **KernighanLinGraphBisector** : Bisects a given graph into two balanced node clusters using the Kernighan Lin algorithm . Wrapper for ``networkx.algorithms.community.kernighan_lin.kernighan_lin_bisection``. We use it to bisect the graph represented by cooccurrence matrix of labels, a hypothesis which was a part of my bachelor's thesis.
* **SphericalKMeansPartitioner** : Partition the data points according to the spherical-k-means algorithm.

#### src/classifiers.py 

Contains various wrappers and implementations for classifiers used in internal or leaf nodes for both ``LabelTree`` and ``ClusterTree``.

* **OVAMultiLabelClassifier** : Takes a ``base_classifier``, which is the instance of a ``scikit-learn`` classifier, and constructs a one-vs-all/binary relevance multilabel classifier, handling edge cases.
* **BinaryLinearSVCWrapper** : Wrapper for sklearn.svm.LinearSVC. Implements ``predict_proba`` method, required for usage with OVAMultiLabelClassifier. The method was removed from the scikit implementation, due to theoretical concerns.
* **ClosestMeanClassifier** : Multiclass classifier that maintains a mean for each class, and uses this to predict the class of a novel sample. Used by CraftML.
* **MeanLabelVectorClassifier** : A trivial multilabel classifier that always returns the mean label vector of its training data. Used by CraftML.

#### src/nn_classifiers.py 

Separate file to avoid importing tensorflow. 

* **MultiLabelMLP** : A wrapper for a simple multilayer dense network, defined in Keras. Default values for the parameters are a mix of those recommended in [this paper][4], and common choices.


#### src/craftml.py

**CraftMLEnsemble**

Wrapper class for a ClusterTree ensemble, as described in CraftML. All parameters are set to default values as per Craftml[].

*Parameters* :

* num_trees : Number of trees
* projector : Class implementing constructor ``__init__(input_dim, output_dim)`` and method ``project_data(x_mat)`` like ``HashingTrickProjector``.
* partitioner : Refer ClusterTree.
* leaf_classifier : Refer ClusterTree.
* internal_classifier : Refer ClusterTree.
* stopping_condition : Refer ClusterTree.
* max_proj_dim : The output_dim of the projector is taken as ``min(input_dim,max_proj_dim)``.


> fit(x_mat,y_mat)

Instantiates ``num_trees`` ``ClusterTree``s with 2 random projectors for each. The ClusterTree is passed the projected label matrix for clustering, and the projected features for classification, with the original label matrix as targets.

> predict_proba(x_tst)

Returns the mean of the predictions of the ClusterTrees.

> fit_predict_proba(x_mat,y_mat,x_tst)

A hack - ``fit`` and ``predict_proba`` rolled into one, keeping just one tree at a time - in the interest of memory efficiency.

**HashingTrickProjector**

Implementation of the "hashing trick" projection described in [this paper][5] and utilized by [CraftML][3].

> project_data(x_mat)

* x_mat : A numpy array or scikit sparse array.

#### src/utils.py

> generate_parabel_label_representations(x_mat,y_mat)

Generate representations for labels as defined in [Parabel][2]. The representation of a label is nothing but the mean of the feature vectors of the points in the support of a label.

> generate_cooccurrence_matrix(y_mat)

Generate a (symmetric) LxL array where the (i,j)th entry is the number of common examples between the ith and jth labels.

### Resources and References

* [An extensive experimental comparison of methods for multi-label learning](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.710.3585&rep=rep1&type=pdf)
* [The Extreme Classification Repository: Multi-label Datasets & Code](http://manikvarma.org/downloads/XC/XMLRepository.html)
* [Effective and Efficient Multilabel Classification in Domains with Large Number of Labels][1]
* [Parabel: Partitioned Label Trees for Extreme Classification with Application to Dynamic Search Advertising][2]
* [CRAFTML, an Efficient Clustering-based Random Forest for Extreme Multi-label Learning][3]
* [Large-scale Multi-label Text Classification - Revisiting Neural Networks][4]
* [Feature Hashing for Large Scale Multitask Learning][5]


[1]: http://lpis.csd.auth.gr/publications/tsoumakas-mmd08.pdf "Effective and Efficient Multilabel Classification in Domains with Large Number of Labels"

[2]: http://manikvarma.org/pubs/prabhu18b.pdf "Parabel: Partitioned Label Trees for Extreme Classification with Application to Dynamic Search Advertising"

[3]: http://proceedings.mlr.press/v80/siblini18a.html "CRAFTML, an Efficient Clustering-based Random Forest for Extreme Multi-label Learning"

[4]: https://arxiv.org/abs/1312.5419 "Large-scale Multi-label Text Classification - Revisiting Neural Networks"

[5]: http://alex.smola.org/papers/2009/Weinbergeretal09.pdf "Feature Hashing for Large Scale Multitask Learning"
