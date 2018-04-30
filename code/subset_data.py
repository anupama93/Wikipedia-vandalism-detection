#!/usr/bin/env python
import numpy as np
import sklearn
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


if __name__ == "__main__":
	
	#-------------------------------------- reading file to make feature matrix ------------------------
	
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
	
	#-------------------------------------------  dividing data into train and validation set ----------------------
		
	train_input = feature_matrix[0:int(len(feature_matrix)*0.7)]
	validation_input = feature_matrix[int(len(feature_matrix)*0.7):]
	train_output = output[0:int(len(output)*0.7)]
	validation_output = output[int(len(output)*0.7):]
	
	#----------------------------------------- seperating metadata features ------------------------
	
	metadata_validation=validation_input[:,0:4]
	metadata_train= train_input[:,0:4]
	
	#----------------------------------------- seperating language features ------------------------
	
	lang_validation=validation_input[:,16:28]
	lang_train= train_input[:,16:28]
	
	#-------------------------------------------- seperating text features -------------------------------
	
	text_validation=validation_input[:,5:15]
	text_train= train_input[:,5:15]
	
	
	#------------------------------------ metadata + language features -----------------------------
	
	met_lang_validation=np.concatenate((metadata_validation, lang_validation), axis=1)
	met_lang_train=np.concatenate((metadata_train, lang_train), axis=1)
	
	#------------------------------------metadata + text features -----------------------------------
	
	met_text_validation=np.concatenate((metadata_validation, text_validation), axis=1)
	met_text_train=np.concatenate((metadata_train, text_train), axis=1)
	
	#-------------------------------------- text + language features -----------------------------------

	text_lang_validation=np.concatenate((text_validation, lang_validation), axis=1)
	text_lang_train=np.concatenate((text_train, lang_train), axis=1)
	
	
	#-------------------------------------- Random Forest Model ------------------------------------
	
	clf=RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=10, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
	
	
	
	#----------------------------------- fit and observation (graph) with metadata features-------------------------------
	clf.fit(metadata_train,train_output)
	predicted_output_metadata = clf.predict_proba(metadata_validation)

	precision_metadata, recall_metadata, thresholds = precision_recall_curve(validation_output, predicted_output_metadata[:, 1])

	plt.step(precision_metadata, recall_metadata, alpha=0.9,where='post')
	
	
	#----------------------------------- fit and observation (graph) with language features-------------------------------
	
	clf.fit(lang_train,train_output)
	predicted_output_lang = clf.predict_proba(lang_validation)

	precision_lang, recall_lang, thresholds = precision_recall_curve(validation_output, predicted_output_lang[:, 1])

	plt.step(precision_lang, recall_lang, alpha=0.9,where='post')
	
	#----------------------------------- fit and observation (graph) with text features-------------------------------
	
	clf.fit(text_train,train_output)
	predicted_output_text = clf.predict_proba(text_validation)

	precision_text, recall_text, thresholds = precision_recall_curve(validation_output, predicted_output_text[:, 1])

	plt.step(precision_text, recall_text, alpha=0.9,where='post')
	
	
	#--------------------------- fit and observation (graph) with text + language features-------------------------------
	
	clf.fit(text_lang_train,train_output)
	predicted_output_text_lang = clf.predict_proba(text_lang_validation)

	precision_text_lang, recall_text_lang, thresholds = precision_recall_curve(validation_output, predicted_output_text_lang[:, 1])

	plt.step(precision_text_lang, recall_text_lang, alpha=0.9,where='post')
	
	
	#-------------------------- fit and observation (graph) with metadata +language features-------------------------------
	
	clf.fit(met_lang_train,train_output)
	predicted_output_met_lang = clf.predict_proba(met_lang_validation)

	precision_met_lang, recall_met_lang, thresholds = precision_recall_curve(validation_output, predicted_output_met_lang[:, 1])

	plt.step(precision_met_lang, recall_met_lang, alpha=0.9,where='post')
	
	
	
	#---------------------------- fit and observation (graph) with metadata + text features-------------------------------
	
	clf.fit(met_text_train,train_output)
	predicted_output_met_text = clf.predict_proba(met_text_validation)

	precision_met_text, recall_met_text, thresholds = precision_recall_curve(validation_output, predicted_output_met_text[:, 1])

	plt.step(precision_met_text, recall_met_text, alpha=0.9,where='post')
	
	
	#----------------------------------- fit and observation (graph) with all features-------------------------------
	clf.fit(train_input,train_output)
	predicted_output = clf.predict_proba(validation_input)

	precision, recall, thresholds = precision_recall_curve(validation_output, predicted_output[:, 1])

	plt.step(precision, recall, alpha=0.9,where='post')
	
	
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-Recall curve')
	plt.legend(['Metadata', 'Language', 'Text', 'Text+Language', 'Metadata+Language', 'Metadata+Text', 'All'], loc='upper right')
	
	plt.show()
	
	
