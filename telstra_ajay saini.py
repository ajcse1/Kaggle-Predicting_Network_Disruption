import pandas as pd
import numpy as np
import random
from collections import Counter
import constants as cons
import time
from sklearn import cross_validation
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


########################################################################################################################
#                                      Pre-Processing                                                                  #
########################################################################################################################
start_time = time.time()

train = pd.read_csv('data/train.csv', index_col = 'id')
test = pd.read_csv('data/test.csv', index_col = 'id')
train['fault_severity'] = train['fault_severity'].apply(lambda x: int(x)) # to make sure fault_severity is of type integer

data = pd.concat([train, test], axis = 0).fillna('predict!')
data = data.sort_index()

event_type = pd.read_csv('data/event_type.csv')
log_feature = pd.read_csv('data/log_feature.csv')
resource_type = pd.read_csv('data/resource_type.csv')
severity_type = pd.read_csv('data/severity_type.csv')

id_order = severity_type.id.values

event_type['event_type'] = event_type['event_type'].apply(lambda x: int(x.split(' ')[1]))
log_feature['log_feature'] = log_feature['log_feature'].apply(lambda x: int(x.split(' ')[1]))
resource_type['resource_type'] = resource_type['resource_type'].apply(lambda x: int(x.split(' ')[1]))
severity_type['severity_type'] = severity_type['severity_type'].apply(lambda x: int(x.split(' ')[1]))
data['location'] = data['location'].apply(lambda x: int(x.split(' ')[1]))

events = pd.get_dummies(event_type['event_type'], prefix = 'e')
events = pd.concat([event_type['id'], events], axis = 1)
events = events.groupby(['id']).sum()

logs = pd.get_dummies(log_feature['log_feature'], prefix = 'l')
logs = logs.multiply(log_feature['volume'].values, axis = 0)
logs = pd.concat([log_feature['id'], logs], axis = 1)
logs = logs.groupby(['id']).sum()

resources = pd.get_dummies(resource_type['resource_type'], prefix = 'r')
resources = pd.concat([resource_type['id'], resources], axis = 1)
resources = resources.groupby(['id']).sum()

severity = pd.get_dummies(severity_type['severity_type'], prefix = 's')
severity = pd.concat([severity_type['id'], severity], axis = 1)
severity = severity.groupby(['id']).sum()

merge_data = pd.concat([data, events, resources, severity, logs], axis = 1)
merge_data = merge_data.reindex(id_order)
merge_data['intra-location num'] = merge_data.groupby('location').cumcount()
merge_data['intra-location fraction'] = merge_data.groupby('location')['intra-location num'].apply(lambda x: x / (x.max() + 1))
merge_data.drop('intra-location num', axis = 1, inplace = True)
merge_data.to_csv('processed_data.csv')

print("--- Pre-processing took %s seconds ---" % (time.time() - start_time))

########################################################################################################################
#                                      Model Building                                                                  #
########################################################################################################################

data = pd.read_csv('processed_data.csv', index_col='id')
train = data.loc[data.fault_severity != 'predict!'].copy()
test = data.loc[data.fault_severity == 'predict!'].copy()
del data
train.loc[:, 'fault_severity'] = train.loc[:, 'fault_severity'].apply(lambda x: int(float(x)))
train_labels = train.fault_severity
test_id = test.index
test.drop('fault_severity', axis=1, inplace=True)
train.drop('fault_severity', axis=1, inplace=True)
train_values = train.values


        #-------------------------------------Random Forest--------------------------------------------#

start_time = time.time()
preds = []  # list to store predictions
rf = RandomForestClassifier(criterion="gini", n_estimators=10, warm_start=False)
num_models = 10  # number of random forest predictions to create

for i in xrange(num_models):
    rf.fit(train,train_labels)
    predicted = cross_validation.cross_val_predict(rf, train, train_labels, cv=10)
    prediction=rf.predict_proba(test)
    prediction = pd.DataFrame(prediction, columns=['predict_' + str(int(x)) for x in set(train_labels)])
    prediction.set_index(test_id, inplace=True)
    prediction.reset_index(inplace=True)
    preds.append(prediction)

accumulative_predictions = pd.concat([i for i in preds], axis=0)
average_prediction = accumulative_predictions.groupby('id').mean()
average_prediction.to_csv('RFPred.csv')

print("--- Random Forest took %s seconds ---" % (time.time() - start_time))


        # -------------------------------------Neural Network--------------------------------------------#

start_time = time.time()
preds = []  # list to store predictions
nn = MLPClassifier(hidden_layer_sizes=(50, 20,), activation='logistic', max_iter=500, alpha=1e-4, solver='sgd',
                   verbose=True, tol=0.0001, learning_rate_init=.001)
num_models = 5  # number of neural network predictions to create

for i in xrange(num_models):
    nn.fit(train,train_labels)
    predicted = cross_validation.cross_val_score(nn, train,train_labels, cv=5,verbose=1)
    prediction=nn.predict_proba(test)
    prediction = pd.DataFrame(prediction, columns=['predict_' + str(int(x)) for x in set(train_labels)])
    prediction.set_index(test_id, inplace=True)
    prediction.reset_index(inplace=True)
    preds.append(prediction)

accumulative_predictions = pd.concat([i for i in preds], axis=0)
average_prediction = accumulative_predictions.groupby('id').mean()
average_prediction.to_csv('NNPred.csv')

print("--- Neural Network took %s seconds ---" % (time.time() - start_time))

        # -------------------------------------XGBoost--------------------------------------------#

start_time = time.time()


num_rounds_dict = {}  # for storing how many rounds to run xgboost for each optimal eta value
max_rounds = 10000
random_seed = 0


def objective(space):
    global random_seed
    global scoring_rounds
    scores = []
    num_rounds = []
    space['objective'] = 'multi:softprob'  # used for multiclass classification
    space['num_class'] = 3  # 3 class classification problem
    space['eval_metric'] = 'mlogloss'  # how is the machine success evaluated
    for i in xrange(scoring_rounds):
        dtrain, dval, ytrain, yval = train_test_split(train_values, train_labels, test_size=0.2,
                                                      random_state=random_seed)
        random_seed += 1
        dtrain = xgb.DMatrix(dtrain, label=ytrain)
        dval = xgb.DMatrix(dval, yval)
        evallist = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(space, dtrain, max_rounds, evallist, early_stopping_rounds=120)
        scores.append(model.best_score)  # append the best score xgboost got on this data split
        num_rounds.append(model.best_iteration)  # how many xgboost rounds was the best score achieved at
        print('Model best score', model.best_score)
    num_rounds_dict[space['eta']] = np.mean(num_rounds)
    return {'loss': np.mean(scores), 'status': STATUS_OK}  # return the average score


space = {
    'eta': hp.uniform('eta', 0.06, 0.2),  # learning rate
    'max_depth': hp.quniform('max_depth', 5, 9, 1),  # max depth of a tree
}

# the following will average a number of xgboost models

num_models = 2  # number of xgboost predictions to create
optimization_rounds = 3  # number of different parameter selections
scoring_rounds = 10  # number of rounds to score per parameter selection

preds = []  # list to store predictions

for i in xrange(num_models):
    best = fmin(fn=objective,  # function to minimize
                space=space,  # parameters to optimize
                algo=tpe.suggest,
                max_evals=optimization_rounds)
    dtest = xgb.DMatrix(test)
    train = xgb.DMatrix(train_values, train_labels)
    best['objective'] = 'multi:softprob'
    best['num_class'] = 3
    best['eval_metric'] = 'mlogloss'
    model = xgb.train(best, train, int(num_rounds_dict[best['eta']]))
    prediction = model.predict(dtest)
    prediction = pd.DataFrame(prediction, columns=['predict_' + str(int(x)) for x in set(train_labels)])
    prediction.set_index(test_id, inplace=True)
    prediction.reset_index(inplace=True)
    preds.append(prediction)

accumulative_predictions = pd.concat([i for i in preds], axis=0)
average_prediction = accumulative_predictions.groupby('id').mean()
average_prediction.to_csv('XGBPred.csv')

print("--- XGBoost took %s seconds ---" % (time.time() - start_time))


# -------------------------------------Ensembling--------------------------------------------#

def combine_models_multi(files, newfile):
    f1 = pd.read_csv(files[0])
    f = pd.DataFrame(f1,copy=True)
    nf = len(files)
    for i in range(1,nf):
        f1 = pd.read_csv(files[i])
        f['predict_0'] += f1['predict_0']
        f['predict_1'] += f1['predict_1']
        f['predict_2'] += f1['predict_2']
    f['predict_0'] /= nf
    f['predict_1'] /= nf
    f['predict_2'] /= nf
    print round(sum(f[['predict_0','predict_1','predict_2']].sum()),0)==f.shape[0]
    f.to_csv(newfile,index=False)

comb = ['%s.csv'%i for i in ['RFPred','XGBPred']]
combine_models_multi(comb,'ensemble.csv')
