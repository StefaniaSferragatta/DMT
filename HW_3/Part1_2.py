# utilities
import pandas as pd
import jsonlines
import json
import pickle
import numpy as np
import ast
from csv import DictReader

# model training
#for binary classification
from sklearn.linear_model import LogisticRegression
#for hyperparams tuning
from sklearn.model_selection import RandomizedSearchCV
#for the confusion matrix
from sklearn import metrics

#visualization 
import seaborn as sns
import matplotlib.pyplot as plt

"""Firstly convert the jsonl files into dataframe and save it into .csv file"""
# DEV SET 
# df = pd.read_json(r'emb_dev.jsonl')
# export_csv = df.to_csv(r'emb_dev.csv', index = None, header=True)

# TEST SET 
# df1 = pd.read_json(r'emb_test.jsonl')
# export_csv = df1.to_csv(r'emb_test.csv', index = None, header=True)

# TRAIN SET 
# df2 = pd.read_json(r'emb_train.jsonl',lines=True)
# export_csv = df2.to_csv(r'emb_train.csv', index = None, header=True)


"""LOAD THE DATA"""

dev_df = pd.read_csv('emb_dev.csv')
train_df = pd.read_csv('emb_train.csv')
test_df = pd.read_csv('emb_test.csv')

''' Method to modify the dataset taking only the usefull fields 'output' and 'claim_embedding' and convert the output's values into binary values for the classification
  Input: dataframe to modify
  Output: modified dataframe
'''
def modify_df(df):  
    #delete the useless columns
    del df['id']
    del df['input']
    
    list_item=[]
    list_values=[]
    
    #extract the items from the output column
    series_output = pd.DataFrame(df['output'])
    for item in series_output['output']:
        #to convert the string of list into list of dict
        list_item=ast.literal_eval(item)
        # change values to represent labels as 0 ("REFUTES") and 1 ("SUPPORTS") and add them to a list
        for i in list_item:
            if (i['answer'] == 'REFUTES'):
                list_values.append(0)
            else:
                list_values.append(1)

    #create a new column of the df with the list of zeros and ones 
    df['labels'] = list_values
    #delete the output column
    del df['output']
    return df

# obtain the dataset processed 
dev = modify_df(dev_df)

# obtain the dataset processed 
train = modify_df(train_df)


"""For training the classification model, firstly we split the datasets:
- train_set divided in:
    - x_train (embeddings vectors)  
    - y_train (label) 
- dev_set divided in:
    - x_dev (embeddings vectors), 
    - y_dev (label)
- test_set becames x_test (embeddings vectors)
"""
#TRAIN SET
x_train_emb = train['claim_embedding']
#convert string to list 
x_train = [n.strip('][').split(', ') for n in x_train_emb]
y_train = train.labels

#DEV SET
x_dev_emb = dev['claim_embedding']
#convert string to list 
x_dev = [n.strip('][').split(', ') for n in x_dev_emb]
y_dev = dev.labels

#TEST SET
test_set = test_df.claim_embedding
#convert string to list 
x_test = [n.strip('][').split(', ') for n in test_set]

""" LR Classifier """

''' Method for tuning the hyperparameters
  Input: splitted train set into x and y for the fit of the RandomizedSearchCV
  Output: best params to use for the KNN
'''
def tuning(x_train,y_train):
    # define the parameter values that should be searched
    penalty=['l1', 'l2', 'elasticnet', 'none']
    dual=[True,False]
    C_range=list(np.arange(1,5,0.5))
    solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    multi_class=['auto', 'ovr', 'multinomial']

    # save the "parameter grid"
    param_grid = dict(penalty=penalty,dual=dual,C=C_range,solver=solver,\
                     multi_class=multi_class)
    print('Params grid: ',param_grid) #need this for the report

    #define the classification model chosen
    model = LogisticRegression()
    rand = RandomizedSearchCV(model, param_grid, cv=5, scoring='accuracy', n_iter=10, random_state=5, n_jobs=-1)
    rand.fit(x_train,y_train)
    rand.cv_results_
    
    # examine the best model
    print('Rand. Best Score: ', rand.best_score_)
    #save the optimize parameters
    best_param = rand.best_params_
    #return the tuning params for the model
    return best_param

params = tuning(x_train,y_train) #dict of best parameters for the classifier

''' Method for training the model using the LogisticRegression() as binary classifier. After the training, the model is saved into a pickle file
  Input: dictionary of the tuned parameters, train set splitted in feature and target (x_train, y_train)
'''
def classifier(params, x_train, y_train):
    #train the model using the optimized params obtained in the tuning
    lr = LogisticRegression(solver=params['solver'],penalty=params['penalty'],C=params['C'],\
    multi_class=params['multi_class'],dual=params['dual'])

    #fit the model
    lr.fit(x_train, y_train)
    
    # save the model to disk
    filename = 'LR.sav'
    pickle.dump(lr, open(filename, 'wb'))

classifier(params, x_train, y_train)

# load the trained model from disk
lr = pickle.load(open('LR.sav', 'rb'))

# make class predictions for the dev set, we need this to evaluate the model
y_pred_class_lr = lr.predict(x_dev)


''' Function for the evaluation of the model. Using the metrics function from the library sklearn, here we compute the accuracy_score, the confusion_matrix and the precision and recall of the targets 'SUPPORTS', 'REFUTES'.
  Input: y_dev, predicted class
  Output: accuracy score,confusion matrix,precision score,recall score
'''
def evaluation(y_dev,y_pred_class):    
    # compute the accuracy 
    accuracy = metrics.accuracy_score(y_dev, y_pred_class)
    
    #build the confusion matrix and plot it
    confusion = metrics.confusion_matrix(y_dev, y_pred_class)
                #[row, column]
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    
    # visualize Confusion Matrix
    sns.heatmap(confusion,annot=True,fmt="d") 
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # compute the precision and the recall on the label and print them
    target_names = ['SUPPORTS', 'REFUTES']
    print(metrics.classification_report(y_dev, y_pred_class, target_names=target_names))
    
    return accuracy,confusion

results = evaluation(y_dev,y_pred_class_lr)

print('Accuracy value: ',results[0])
conf_matrix = results[1]

conf_matrix

print('F1 value: ', metrics.f1_score(y_dev, y_pred_class_lr))

"""For the chosen classifier, get predictions for the official test set associated to the best hyperparameter configuration"""

# get predictions for the official test set 
pred_test = lr.predict(x_test)

with open('emb_test.csv', 'r') as read_obj:
    # pass the file object to DictReader() to get the DictReader object
    dict_reader = DictReader(read_obj)
    # get a list of dictionaries from dict_reader
    json_test = list(dict_reader)

    #delete the useless fields    
for i in range(len(json_test)):
    del json_test[i]['claim_embedding'], json_test[i]['input']

#add the field 'output' with the value that corresponds to the prediction
for index in range(len(json_test)): 
    if pred_test[index]==1:
        json_test[index]['output']= [{'answer':"SUPPORT"}]
    else:
        json_test[index]['output']= [{'answer':"REFUTES"}]

"""Put the predictions in a file named “test_set_pred_2.jsonl” """

# save the prediction into a file
with jsonlines.open('test_set_pred_1.jsonl', mode = 'w') as writer:
    writer.write(json_test)
