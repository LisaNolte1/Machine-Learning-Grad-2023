from random import seed
from random import randrange
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder

def load_csv(filename, trainfile):
    columns = ["TX_FRAUD", "TX_TS", "TX_AMOUNT", "TRANSACTION_GOODS_AND_SERVICES_AMOUNT", "TRANSACTION_CASHBACK_AMOUNT", "CARD_EXPIRY_DATE", "CARD_BRAND", "TRANSACTION_TYPE", "TRANSACTION_STATUS", "FAILURE_CODE", "TRANSACTION_CURRENCY", "CARD_COUNTRY_CODE", "IS_RECURRING_TRANSACTION", "CARDHOLDER_AUTH_METHOD"]

    # Load your datasets
    dataset = pd.read_csv(os.path.join(filename, trainfile))
    customer_info = pd.read_csv(os.path.join(filename, 'customers.csv'))
    merchant_info = pd.read_csv(os.path.join(filename, 'merchants.csv'))
    terminal_info = pd.read_csv(os.path.join(filename, 'terminals.csv'))

    # Merge datasets
    merged_dataset = dataset.merge(customer_info, on='CUSTOMER_ID', how='left')
    merged_dataset = merged_dataset.merge(merchant_info, on='MERCHANT_ID', how='left')
    merged_dataset = merged_dataset.merge(terminal_info, on='TERMINAL_ID', how='left')

    # Select the desired columns from merged_dataset
    selected_columns = merged_dataset[columns]

    # Apply one-hot encoding to categorical columns in the selected data
    categorical_columns = ['CARD_BRAND', 'TRANSACTION_TYPE', 'TRANSACTION_STATUS', 'TRANSACTION_CURRENCY', 'TX_FRAUD']
    encoder = OneHotEncoder(sparse=False, drop='first')
    one_hot_encoded = encoder.fit_transform(selected_columns[categorical_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    # Combine the one-hot encoded data with the selected columns
    data = selected_columns.drop(categorical_columns, axis=1)
    data = pd.concat([data, one_hot_df], axis=1)

    return data

# Convert string column to float
def str_column_to_float(dataset):
    for column in dataset.select_dtypes(include=['number']):
        dataset[column] = dataset[column].astype(float)
    return dataset

# Convert string column to integer
def str_column_to_int(dataset):
    for column in dataset.select_dtypes(include=['number']):
        dataset[column] = dataset[column].astype(int)
    return dataset  

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset.values)
    n = len(dataset_copy)
    fold_size = n // n_folds  # Calculate the fold size

    if n < n_folds:
        raise ValueError("Number of folds is greater than the number of data points.")

    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)

    # If there are remaining data points, add them to the last fold
    while dataset_copy:
        dataset_split[-1].append(dataset_copy.pop())

    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Make a prediction with weights
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        try:
            activation += weights[i + 1] * row[i]
        except TypeError:
            pass
    return 1.0 if activation >= 0.0 else 0.0

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
    if os.path.exists("output.csv"):
        weights = pd.read_csv('output.csv').values
    else:
        weights = [0.0 for _ in range(len(train[0]))]
    
    for _ in range(n_epoch):
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                try:
                    weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
                except TypeError:
                    pass
    
    df = pd.DataFrame(weights)
    df.to_csv('output.csv', index=False)
    return weights

# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron(train, test, l_rate, n_epoch):
    predictions = list()
    weights = train_weights(train, l_rate, n_epoch)
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return(predictions)

def main():
    trainFiles = [x for x in os.listdir(os.path.join(os.getcwd(), "perceptron\\src\\data")) if x.startswith("transactions_train")]
    for fileName in trainFiles:
        seed(1)
        # Load and merge data
        folder_path = os.path.join(os.getcwd(), "perceptron\\src\\data")
        dataset = load_csv(folder_path, fileName)
        dataset['TX_TS'] = pd.to_datetime(dataset['TX_TS'])
    
        str_column_to_float(dataset)
        str_column_to_int(dataset)
	    # evaluate algorithm
        n_folds = 2
        l_rate = 0.01
        n_epoch = 500
        scores = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
        #TX_ID = testSet['TX_ID']
        #TX_FRAUD = predictions
        #save those to csv
        # [TX_ID, TX_FRAUD]

if __name__ == "__main__":
	main()