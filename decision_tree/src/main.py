import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

from sklearn import metrics

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os

def check_files():
    for dirname, _, filenames in os.walk('src/data'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

def load_csv(path, file):
    #load datafile        
    dataset = pd.read_csv(path + file)
    
    #load other files
    customer_info = pd.read_csv(path + 'customers.csv')
    merchant_info = pd.read_csv(path + 'merchants.csv')
    terminal_info = pd.read_csv(path + 'terminals.csv')
    
    #merge with datafile
    merged_dataset = dataset.merge(customer_info, on='CUSTOMER_ID', how='left')
    merged_dataset = merged_dataset.merge(merchant_info, on='MERCHANT_ID', how='left')
    merged_dataset = merged_dataset.merge(terminal_info, on='TERMINAL_ID', how='left')
    
    return merged_dataset

def get_XY(data, columns):
    #features(deciding dactors X
    X = data[columns]
    
    #colomn to predict Y
    Y = data['TX_FRAUD']
    
    return X, Y

def create_model(X , Y, leaf_node, columns):
    model_tree = DecisionTreeClassifier(max_leaf_nodes=leaf_node, class_weight='balanced')
    model_tree.fit(X , Y)
    
    #Create the figure
    plt.figure(figsize=(20,10))

    #Create the tree plot
    plot_tree(model_tree,
           feature_names = columns, #Feature names
           class_names = ["0","1"], #Class names
           rounded = True,
           filled = True)

    plt.show()
    
    return model_tree

def get_best_leaf_node(x, y):
    leaf_node = 8
    diff_score = 10
    for num_leaf_node in range(2,16):
        model_tree = DecisionTreeClassifier(max_leaf_nodes=num_leaf_node, class_weight='balanced')
        kfold_scores = cross_validate(model_tree,
                                  x,
                                  y,
                                  cv=5,
                                  scoring=cv_roc_auc_scorer,
                                  return_train_score=True)
        
        # Find average train and test score
        train_auc_avg = np.mean(kfold_scores['train_score'])
        test_auc_avg = np.mean(kfold_scores['test_score'])
        
        print("Nodes:{}, Train:{:.4f}, Valid:{:.4f}, Diff:{:.4f}".format(num_leaf_node,
                                                                     train_auc_avg,
                                                                     test_auc_avg,
                                                                     train_auc_avg-test_auc_avg))
        
        if((train_auc_avg - test_auc_avg) < diff_score ):
            diff_score = train_auc_avg - test_auc_avg
            leaf_node = num_leaf_node
            
    print("Nodes:{}, Diff:{:.4f}".format(leaf_node, diff_score))
    return leaf_node

def cv_roc_auc_scorer(model, X, y): 
    return metrics.roc_auc_score(y, model.predict(X))

def main():
    #file path
    path = 'src/data/'
    
    #feature names
    columns = [ "TX_AMOUNT", "TRANSACTION_GOODS_AND_SERVICES_AMOUNT"
               , "TRANSACTION_CASHBACK_AMOUNT", "MCC_CODE", "x_customer_id"
               , "y_customer_id", "x_terminal_id", "y_terminal__id", "ANNUAL_TURNOVER_CARD"
               , "AVERAGE_TICKET_SALE_AMOUNT", "PAYMENT_PERCENTAGE_FACE_TO_FACE", "PAYMENT_PERCENTAGE_ECOM"
               , "DEPOSIT_PERCENTAGE"]
    
    #load training csv
    train_data = load_csv(path, 'transactions_train.csv')
    
    #number of frauds
    print(train_data.groupby('TX_FRAUD').size())
    
    #create x and y
    X , Y = get_XY(train_data, columns)
    
    #Get the optimum leaf nodes
    leaf_node = get_best_leaf_node(X , Y)
    
    #create  final model
    model = create_model(X , Y, leaf_node, columns)
    
    #load test csv
    test_data = load_csv(path, 'transactions_test.csv')
    
    print(test_data.describe())
    
    #Make predictions
    X_test = test_data[columns]
    y_test_pred  = model.predict_proba(X_test)
    
    print("Predictions")
    print(y_test_pred)
    
    #Create new datframe for submission
    TX_ID = test_data.loc[:, 'TX_ID']
    submit_data = {'TX_ID': TX_ID, 'TX_FRAUD': y_test_pred[:, 1]}  
    submit_df = pd.DataFrame(submit_data)
    
    #Write data to a new csv
    submit_df.to_csv(path + '/fraud_detect_tree_submission.csv', index=False)

if __name__ == "__main__":
    main()