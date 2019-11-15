#!/usr/bin/env python
'''
AVCLASS++ propagator
'''

import os
import sys
path = os.path.dirname(os.path.abspath(__file__))
libpath = os.path.join(path, 'lib/')
sys.path.insert(0, libpath)
import argparse
import hashlib
import glob
import copy
import numpy as np
import pandas as pd
from avclass_labeler import guess_hash
from ember import PEFeatureExtractor
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.model_selection import cross_val_score
import pickle
import optuna

def get_file_hash_value(data, hash_type):
    if hash_type == 'md5':
        return hashlib.md5(data).hexdigest()
    elif hash_type == 'sha1':
        return hashlib.sha1(data).hexdigest()
    elif hash_type == 'sha256':
        return hashlib.sha256(data).hexdigest()
    else:
        exit(1)   

def main(args):
    # Prepare labels
    label_hash_list = []
    with open(str(args.labels), 'r') as f:
        for line in f:
            hash_value = line.split('\t')[0].strip()
            label = line.split('\t')[1].strip()

            if label.startswith('SINGLETON'):
                label = 'SINGLETON'

            label_hash_list.append([label, hash_value])

    df = pd.DataFrame(label_hash_list, columns=['label', 'hash'])

    # Prepare features
    hash_features_list = []
    extractor = PEFeatureExtractor()
    hash_type = guess_hash(hash_value)

    for sample in glob.glob(str(args.sampledir) + '/*'):
        with open(sample, 'rb') as f:
            file_data = f.read()
            file_hash_value = get_file_hash_value(file_data, hash_type)
            features = np.array(extractor.feature_vector(file_data), dtype=np.float32)

        hash_features_list.append([file_hash_value, features])

    df2 = pd.DataFrame(hash_features_list, columns=['hash', 'features'])

    # Merge data frames; Samples not included in the .labels file are ignored
    df['features'] = df2['features']
    del df2

    X = df['features'].tolist()
    y = df['label'].tolist()

    # Encode labels
    le = LabelEncoder()
    le.fit(y)
    index_singleton = list(le.classes_).index('SINGLETON')
    y_encoded = le.transform(y)
    y_encoded = [x if x!= index_singleton else -1 for x in y_encoded]

    # Label propagation / spreading
    if args.opt == False:
        def objective(trial):
            classifier_name = trial.suggest_categorical('classifier', ['LabelPropagation', 'LabelSpreading'])
            if classifier_name == 'LabelPropagation':
                params = {
                    'kernel': trial.suggest_categorical('kernel', ['knn', 'rbf']),
                    'gamma': trial.suggest_loguniform('gamma', 1e-8, 1e+8),
                    'n_neighbors': trial.suggest_int('n_neighbors', 2, 10),
                    'max_iter': trial.suggest_int('max_iter', 100, 1000),
                    'tol': trial.suggest_loguniform('alpha', 1e-8, 1.0),
                }
                clf = LabelPropagation(**params)
            if classifier_name == 'LabelSpreading':
                params = {
                    'kernel': trial.suggest_categorical('kernel', ['knn', 'rbf']),
                    'gamma': trial.suggest_loguniform('gamma', 1e-8, 1e+8),
                    'n_neighbors': trial.suggest_int('n_neighbors', 2, 10),
                    'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
                    'max_iter': trial.suggest_int('max_iter', 100, 1000),
                    'tol': trial.suggest_loguniform('alpha', 1e-8, 1.0),
                }
                clf = LabelSpreading(**params)

            score = cross_val_score(clf, X, y_encoded, n_jobs=-1, cv=2)
            accuracy = score.mean()

            clf.fit(X, y_encoded)

            if not os.path.exists("models"):
                os.mkdir("models")

            with open('models/{}.pickle'.format(trial.number), 'wb') as fout:
                pickle.dump(clf, fout)

            return accuracy

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        trial = study.best_trial
        with open('models/{}.pickle'.format(study.best_trial.number), 'rb') as fin:
            clf = pickle.load(fin)

    else:
        clf = LabelPropagation()
        clf.fit(X, y_encoded)

    # Decode labels
    y_encoded2 = clf.predict(X)
    y2 = [le.classes_[x] for x in y_encoded2]

    # Replace SINGLETON labels with predicted labels
    for i in range(len(y)):
        if y[i] == 'SINGLETON':
            y[i] = y2[i] 

    # Output results with .labels file format
    hash_list = df['hash'].tolist()

    if args.results:
        results = args.results
    else:
        results = args.labels.split('.')[0] + '_pr.labels'

    with open(str(results), 'w+') as f:
        for i in range(len(y)):
            f.writelines(hash_list[i] + '\t' + y[i] + '\n')

if __name__=='__main__':
    argparser = argparse.ArgumentParser(prog='avclass_propagator',
        description='''AVCLASS++ propagator''')

    argparser.add_argument('-labels',
        help='existing .labels file generated by AVCLASS labeler')

    argparser.add_argument('-sampledir',
        help='existing directory with malware samples (PE)')

    argparser.add_argument('-results',
        help='new .labels file (new _pr.labels file is output by default)')

    argparser.add_argument('-opt',
        action='store_false',
        help='enable hyperparameter optimization')

    args = argparser.parse_args()

    if not args.labels and not args.sampledir:
        sys.stderr.write('Both of the following 2 arguments is required: '
                          '-labels, -sampledir\n')
        exit(1)

    main(args)
