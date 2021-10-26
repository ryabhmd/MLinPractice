#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train or evaluate a single classifier with its given set of hyperparameters.

Created on Wed Sep 29 14:23:48 2021

@author: lbechberger
"""

import argparse, pickle
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from mlflow import log_metric, log_param, set_tracking_uri

# setting up CLI
parser = argparse.ArgumentParser(description = "Classifier")
parser.add_argument("input_file", help = "path to the input pickle file")
parser.add_argument("-s", '--seed', type = int, help = "seed for the random number generator", default = None)
parser.add_argument("-e", "--export_file", help = "export the trained classifier to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import a trained classifier from the given location", default = None)
parser.add_argument("-m", "--majority", action = "store_true", help = "majority class classifier")
parser.add_argument("-f", "--frequency", action = "store_true", help = "label frequency classifier")
parser.add_argument("--knn", type = int, help = "k nearest neighbor classifier with the specified value of k", default = None)
parser.add_argument("-n", "--naive_bayes", action = "store_true", help = "NaiveBayes classifier", default = None)
parser.add_argument("-r", "--logistic_regression", action = "store_true", help = "logistic_regression classifier", default = None )
parser.add_argument("-t", "--dicision_tree", action = "store_true", help = "dicision_tree classifier", default = None )
parser.add_argument("--random_forest", action = "store_true", help = "random_forest classifier", default = None )
parser.add_argument("--svc", action = "store_true", help = "svc classifier", default = None )
parser.add_argument("--linear_svc", action = "store_true", help = "linear_svc classifier", default = None )
parser.add_argument("-a", "--accuracy", action = "store_true", help = "evaluate using accuracy")
parser.add_argument("-k", "--kappa", action = "store_true", help = "evaluate using Cohen's kappa")
parser.add_argument("--log_folder", help = "where to log the mlflow results", default = "data/classification/mlflow")
args = parser.parse_args()

# load data
with open(args.input_file, 'rb') as f_in:
    data = pickle.load(f_in)

set_tracking_uri(args.log_folder)

if args.import_file is not None:
    # import a pre-trained classifier
    with open(args.import_file, 'rb') as f_in:
        input_dict = pickle.load(f_in)
    
    classifier = input_dict["classifier"]
    for param, value in input_dict["params"].items():
        log_param(param, value)
    
    log_param("dataset", "validation")


else:   # manually set up a classifier
    
    if args.majority:
        # majority vote classifier
        print("    majority vote classifier")
        log_param("classifier", "majority")
        params = {"classifier": "majority"}
        classifier = DummyClassifier(strategy = "most_frequent", random_state = args.seed)
        
    elif args.frequency:
        # label frequency classifier
        print("    label frequency classifier")
        log_param("classifier", "frequency")
        params = {"classifier": "frequency"}
        classifier = DummyClassifier(strategy = "stratified", random_state = args.seed)
        
    
    #K_Neighbors classifier
    elif args.knn is not None:
        print("    {0} nearest neighbor classifier".format(args.knn))
        log_param("classifier", "knn")
        log_param("k", args.knn)
        params = {"classifier": "knn", "k": args.knn}
        standardizer = StandardScaler()
        knn_classifier = KNeighborsClassifier(args.knn, n_jobs = -1)
        classifier = make_pipeline(standardizer, knn_classifier)
        
    #NaiveBayes classifier
    elif args.naive_bayes is not None:
        print (" naive_bayes Classifier")
        log_param("classifier", "naive_bayes")
        params = {"classifier": "naive_bayes"}
        classifier = GaussianNB()
        
    #LinearRegression classifier
    elif args.logistic_regression is not None:
        print(" logistic_regression classifier")
        log_param("classifier", "logistic_regression")
        params = {"classifier": "logistic_regression"}
        classifier = LogisticRegression()
        print ("penalty " +classifier.penalty)
        
    #dicision_tree classifier
    elif args.dicision_tree is not None:
        print("dicision_tree classifier")
        log_param("classifier", "dicision_tree")
        params = {"classifier": "dicision_tree"}
        classifier = DecisionTreeClassifier()
        #print ("leaves count: "+ classifier.ccp_alpha)
        
    #random_forest classifier
    elif args.random_forest is not None:
        print ("random_forest classifier")
        log_param("classifier", "random_forest")
        params = {"classifier" :"random_forest"}
        classifier = RandomForestClassifier(n_estimators=100)
        
    #svc classifier
    elif args.svc is not None:
        print ("svc classifier")
        log_param("classifier", "svc")
        params = {"classifier" :"svc"}
        classifier = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        
    #linear_svc classifier
    elif args.linear_svc is not None:
        print ("linear_svc classifier")
        log_param("classifier", "linear_svc")
        params = {"classifier" :"linear_svc"}
        classifier = SVC(kernel='linear')
        
    
    
    classifier.fit(data["features"], data["labels"].ravel())
    log_param("dataset", "training")

# now classify the given data
prediction = classifier.predict(data["features"])

# collect all evaluation metrics
evaluation_metrics = []
if args.accuracy:
    evaluation_metrics.append(("accuracy", accuracy_score))
if args.kappa:
    evaluation_metrics.append(("Cohen_kappa", cohen_kappa_score))
    
    
# compute and print them
for metric_name, metric in evaluation_metrics:
    metric_value = metric(data["labels"], prediction)
    print("    {0}: {1}".format(metric_name, metric_value))
    log_metric(metric_name, metric_value)
    
# export the trained classifier if the user wants us to do so
if args.export_file is not None:
    output_dict = {"classifier": classifier, "params": params}
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(output_dict, f_out)
        
        