#!/usr/bin/env python
import numpy as np
import sklearn
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


if __name__ == "__main__":
	file = open("wvc2010_features.dat", "r")
	
	feature_matrix = np.zeros((32439,27))
	output = np.zeros((32439))
	row_num = 0
	for line in file:
		
		line_array = line.split(',')
		for i in np.arange(len(line_array)-2):
			feature_matrix[row_num][i] = line_array[i+1]
		output[row_num] = line_array[len(line_array)-1]
		row_num = row_num+1
	
	print('feature matrix made!!!')
	#feature_matrix = sklearn.preprocessing.normalize(feature_matrix, norm='l2', axis=1, copy=True, return_norm=False)
	
	#Defining five fold cross validation
	kf = KFold(n_splits=5)
	
	#Initialising all the required variables
	i=0
	avg_precision_RFC = 0
	avg_precision_MLP = 0
	avg_precision_DTC = 0
	avg_precision_NB = 0
	avg_precision_LR = 0
	avg_precision_ABC = 0
	
	recall_binary_RFC = 0
	recall_binary_MLP = 0
	recall_binary_DTC = 0
	recall_binary_NB = 0
	recall_binary_LR = 0
	recall_binary_ABC = 0
	
	recall_micro_RFC = 0
	recall_micro_MLP = 0
	recall_micro_DTC = 0
	recall_micro_NB = 0
	recall_micro_LR = 0
	recall_micro_ABC = 0
	
	f1_micro_RFC = 0
	f1_micro_MLP = 0
	f1_micro_DTC = 0
	f1_micro_NB = 0
	f1_micro_LR = 0
	f1_micro_ABC = 0
	
	f1_binary_RFC = 0
	f1_binary_MLP = 0
	f1_binary_DTC = 0
	f1_binary_NB = 0
	f1_binary_LR = 0
	f1_binary_ABC = 0

	avg_area_NB = 0
	avg_area_MLP = 0
	avg_area_DTC = 0
	avg_area_RFC = 0
	avg_area_LR = 0
	avg_area_ABC = 0
	
	avg_roc_NB = 0
	avg_roc_MLP = 0
	avg_roc_DTC = 0
	avg_roc_RFC = 0
	avg_roc_LR = 0
	avg_roc_ABC = 0
	
	#Running code for each of the validations
	for train_index, test_index in kf.split(feature_matrix):
		#Collecting train and test data
		train_input, validation_input = feature_matrix[train_index], feature_matrix[test_index]
		train_output, validation_output = output[train_index], output[test_index]
		
		print('loop num: ',i)
		
		#Naive Bayes
		gnb = sklearn.naive_bayes.GaussianNB()
		predicted_output_NB1 = gnb.fit(train_input, train_output).predict(validation_input)
		predicted_output_NB = gnb.fit(train_input, train_output).predict_proba(validation_input)
		average_precision_NB = average_precision_score(validation_output,predicted_output_NB1)
		binary_recall_NB = recall_score(validation_output,predicted_output_NB1, average='binary')
		f1_score_binary_NB =	f1_score(validation_output,predicted_output_NB1, average='binary')
		print('f1_score_binary_NB1: ',f1_score_binary_NB)	
		print('binary_recall_NB1: ',binary_recall_NB)
		print('Precision_NB1: ',average_precision_NB)
		print()
		
		#Random Forest Classifier
		clf=RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=10, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None,  bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
		clf.fit(train_input,train_output)
		predicted_output_RFC = clf.predict_proba(validation_input)
		predicted_output_RFC1 = clf.predict(validation_input)
		average_precision_RFC = average_precision_score(validation_output,predicted_output_RFC1)
		binary_recall_RFC = recall_score(validation_output,predicted_output_RFC1, average='binary')
		f1_score_binary_RFC =	f1_score(validation_output,predicted_output_RFC1, average='binary')
		print('f1_score_binary_RFC: ',f1_score_binary_RFC)	
		print('binary_recall_RFC: ',binary_recall_RFC)
		print('Precision_RFC : ',average_precision_RFC)
		print()
	
		#Logistic Regression
		clf_LR = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=500, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
		clf_LR.fit(train_input,train_output)
		predicted_output_LR = clf_LR.predict_proba(validation_input)
		predicted_output_LR1 = clf_LR.predict(validation_input)
		average_precision_LR1 = average_precision_score(validation_output,predicted_output_LR1)
		binary_recall_LR1 = recall_score(validation_output,predicted_output_LR1, average='binary')
		f1_score_binary_LR1 =	f1_score(validation_output,predicted_output_LR1, average='binary')
		print('f1_score_binary_LR1: ',f1_score_binary_LR1)	
		print('binary_recall_LR1: ',binary_recall_LR1)
		print('Precision_LR1: ',average_precision_LR1)
		print()
	
	
		#Multi Layer Perceptron
		clf_MLP = MLPClassifier(activation='tanh', alpha=0.0000001, batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False,epsilon=0.0001, hidden_layer_sizes=(28,10,2), learning_rate='constant', learning_rate_init=0.001, max_iter=500, momentum=0.9, nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,solver='adam', tol=0.0001,validation_fraction=0.1, verbose=False,warm_start=False)
		clf_MLP.fit(train_input,train_output)
		predicted_output_MLP = clf_MLP.predict_proba(validation_input)
		predicted_output_MLP1 = clf_MLP.predict(validation_input)
		average_precision_MLP1 = average_precision_score(validation_output,predicted_output_MLP1)
		binary_recall_MLP1 = recall_score(validation_output,predicted_output_MLP1, average='binary')
		f1_score_binary_MLP1 =	f1_score(validation_output,predicted_output_MLP1, average='binary')
		print('f1_score_binary_MLP1: ',f1_score_binary_MLP1)	
		print('binary_recall_MLP1: ',binary_recall_MLP1)
		print('Precision_MLP1: ',average_precision_MLP1)
		print()
		
		#Decision Tree classifier
		clf_DTC = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=8, min_samples_leaf=15, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=0.0, class_weight=None, presort=False)
		clf_DTC.fit(train_input,train_output)
		predicted_output_DTC = clf_DTC.predict_proba(validation_input)
		predicted_output_DTC1 = clf_DTC.predict(validation_input)
		average_precision_DTC = average_precision_score(validation_output,predicted_output_DTC1)
		binary_recall_DTC = recall_score(validation_output,predicted_output_DTC1, average='binary')
		f1_score_binary_DTC =	f1_score(validation_output,predicted_output_DTC1, average='binary')
		print('f1_score_binary_DTC: ',f1_score_binary_DTC)	
		print('binary_recall_DTC: ',binary_recall_DTC)
		print('Precision_DTC: ',average_precision_DTC)
		print()
		
		#Adaboost Classifier
		clf_ABC = sklearn.ensemble.AdaBoostClassifier(base_estimator=None, n_estimators=100, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
		clf_ABC.fit(train_input,train_output)
		predicted_output_ABC = clf_ABC.predict_proba(validation_input)
		predicted_output_ABC1 = clf_ABC.predict(validation_input)
		average_precision_ABC = average_precision_score(validation_output,predicted_output_ABC1)
		binary_recall_ABC = recall_score(validation_output,predicted_output_ABC1, average='binary')
		f1_score_binary_ABC =	f1_score(validation_output,predicted_output_ABC1, average='binary')
		print('f1_score_binary_ABC: ',f1_score_binary_ABC)	
		print('binary_recall_ABC: ',binary_recall_ABC)
		print('Precision_ABC : ',average_precision_ABC)
		print()
		
		#Evaluating AUC ROC for each of the classifiers
		roc_auc_NB = roc_auc_score(validation_output, predicted_output_NB1)
		roc_auc_RFC = roc_auc_score(validation_output, predicted_output_RFC1)
		roc_auc_LR = roc_auc_score(validation_output, predicted_output_LR1)
		roc_auc_MLP = roc_auc_score(validation_output, predicted_output_MLP1)
		roc_auc_ABC = roc_auc_score(validation_output, predicted_output_ABC1)
		roc_auc_DTC = roc_auc_score(validation_output, predicted_output_DTC1)
		
		#Evaluating Precision for each of the classifiers		
		precision_NB, recall_NB, _ = precision_recall_curve(validation_output, predicted_output_NB[:, 1])
		precision_RFC, recall_RFC, _ = precision_recall_curve(validation_output, predicted_output_RFC[:, 1])
		precision_MLP, recall_MLP, _ = precision_recall_curve(validation_output, predicted_output_MLP[:, 1])
		precision_DTC, recall_DTC, _ = precision_recall_curve(validation_output, predicted_output_DTC[:, 1])
		precision_ABC, recall_ABC, _ = precision_recall_curve(validation_output, predicted_output_ABC[:, 1])
		precision_LR, recall_LR, _ = precision_recall_curve(validation_output, predicted_output_LR[:, 1])
		
		#Evaluating AUC PR for each of the classifiers
		area_NB = metrics.auc(precision_NB, recall_NB,reorder = True)
		area_RFC = metrics.auc(precision_RFC, recall_RFC,reorder = True)
		area_ABC = metrics.auc(precision_ABC, recall_ABC,reorder = True)
		area_MLP = metrics.auc(precision_MLP, recall_MLP,reorder = True)
		area_DTC = metrics.auc(precision_DTC, recall_DTC,reorder = True)
		area_LR = metrics.auc(precision_LR, recall_LR,reorder = True)

		#Plotting PR curve for each of the classifiers
		plt.step(recall_RFC, precision_RFC, alpha=1,where='post')
		plt.step(recall_DTC, precision_DTC, alpha=1,where='post')
		plt.step(precision_NB, recall_NB, alpha=1,where='post')
		plt.step(precision_MLP, recall_MLP, alpha=1,where='post')
		plt.step(recall_ABC, precision_ABC, alpha=1,where='post')
		plt.step(recall_LR, precision_LR, alpha=1,where='post')
		
		#Plot characteristics
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.title('Precision-Recall curve')
		plt.legend(['Random Forest', 'Decision Tree','Naive Bayes','Multi Layer Perceptron','Adaboost','Logistic Regression'], loc='upper right')
		plt.savefig(str(i)+'.png')
		plt.close()
		
		i=i+1

		#Evaluating all Performance metrics for each of the classifiers for each fold of cross validation to average out later
		avg_precision_RFC += average_precision_RFC
		avg_precision_MLP += average_precision_MLP1
		avg_precision_DTC += average_precision_DTC
		avg_precision_NB += average_precision_NB
		avg_precision_ABC += average_precision_ABC
		avg_precision_LR += average_precision_LR1 
	
		recall_binary_RFC += binary_recall_RFC
		recall_binary_MLP += binary_recall_MLP1
		recall_binary_DTC += binary_recall_DTC
		recall_binary_NB += binary_recall_NB
		recall_binary_ABC += binary_recall_ABC
		recall_binary_LR += binary_recall_LR1
	
		f1_binary_RFC += f1_score_binary_RFC
		f1_binary_MLP += f1_score_binary_MLP1
		f1_binary_DTC += f1_score_binary_DTC
		f1_binary_NB += f1_score_binary_NB
		f1_binary_ABC += f1_score_binary_ABC
		f1_binary_LR += f1_score_binary_LR1
		
		avg_area_NB += area_NB
		avg_area_MLP += area_MLP
		avg_area_DTC += area_DTC
		avg_area_RFC += area_RFC
		avg_area_ABC += area_ABC
		avg_area_LR += area_LR
		
		avg_roc_NB += roc_auc_NB 
		avg_roc_MLP += roc_auc_MLP
		avg_roc_DTC += roc_auc_DTC
		avg_roc_RFC += roc_auc_RFC
		avg_roc_ABC += roc_auc_ABC
		avg_roc_LR += roc_auc_LR
		
		
	#Printing average over the five fold cross validations of each of the performance metrics	
	print('avg_pr_area_RFC',avg_area_RFC/5)
	print('avg_pr_area_MLP',avg_area_MLP/5)
	print('avg_pr_area_DTC',avg_area_DTC/5)
	print('avg_pr_area_NB',avg_area_NB/5)
	print('avg_pr_area_ABC',avg_area_ABC/5)
	print('avg_pr_area_LR',avg_area_LR/5)
	print()		
	
	print('avg_roc_RFC',avg_roc_RFC/5)
	print('avg_roc_MLP',avg_roc_MLP/5)
	print('avg_roc_DTC',avg_roc_DTC/5)
	print('avg_roc_NB',avg_roc_NB/5)
	print('avg_roc_ABC',avg_roc_ABC/5)
	print('avg_roc_LR',avg_roc_LR/5)
	print()
	
	print('avg_precision_RFC',avg_precision_RFC/5)
	print('avg_precision_MLP',avg_precision_MLP/5)
	print('avg_precision_DTC',avg_precision_DTC/5)
	print('avg_precision_NB',avg_precision_NB/5)
	print('avg_precision_ABC',avg_precision_ABC/5)
	print('avg_precision_LR',avg_precision_LR/5)
	print()
	
	print('recall_binary_RFC',recall_binary_RFC/5)
	print('recall_binary_MLP',recall_binary_MLP/5)
	print('recall_binary_DTC',recall_binary_DTC/5)
	print('recall_binary_NB',recall_binary_NB/5)
	print('recall_binary_ABC',recall_binary_ABC/5)
	print('recall_binary_LR',recall_binary_LR/5)
	print()
	
	print('f1_binary_RFC',f1_binary_RFC/5)
	print('f1_binary_MLP',f1_binary_MLP/5)
	print('f1_binary_DTC',f1_binary_DTC/5)
	print('f1_binary_NB',f1_binary_NB/5)
	print('f1_binary_ABC',f1_binary_ABC/5)
	print('f1_binary_LR',f1_binary_LR/5)
	print()