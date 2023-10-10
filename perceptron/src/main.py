from random import seed
from random import randrange
from csv import reader
import pandas as pd
import os

def load_csv(filename):
    data_dict = dict()
    columns = ["TX_ID", "TX_TS", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT", "TRANSACTION_GOODS_AND_SERVICES_AMOUNT", "TRANSACTION_CASHBACK_AMOUNT", "CARD_EXPIRY_DATE", "CARD_DATA", "TRANSACTION_TYPE", "TRANSACTION_STATUS", "TRANSACTION_CURRENCY", "CARD_COUNTRY_CODE", "MERCHANT_ID", "IS_RECURRING_TRANSACTION", "ACQUIRER_ID", "CARDHOLDER_AUTH_METHOD"]

    dataset = pd.read_csv(os.path.join(filename, "transactions_train.csv"))
    customer_info = pd.read_csv(os.path.join(filename, 'customers.csv'))
    merchant_info = pd.read_csv(os.path.join(filename, 'merchants.csv'))
    terminal_info = pd.read_csv(os.path.join(filename, 'terminals.csv'))

    data_dict['dataset'] = dataset

    merged_dataset = dataset.merge(customer_info, on='CUSTOMER_ID', how='left')
    merged_dataset = merged_dataset.merge(merchant_info, on='MERCHANT_ID', how='left')
    merged_dataset = merged_dataset.merge(terminal_info, on='TERMINAL_ID', how='left')

    data_dict['merged_dataset'] = merged_dataset

    # Select the desired columns from merged_dataset
    selected_columns = merged_dataset[columns]

    # Convert the selected columns to a list of lists
    selected_columns_list = selected_columns.values.tolist()

    return selected_columns_list

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset[1:]:
        value = row[column]
        if isinstance(value, str):
            try:
                row[column] = float(value.strip())
            except ValueError:
                pass  # Handle invalid float values
        elif isinstance(value, int):
            # No need to convert, it's already an integer
            pass
        else:
            # Handle other data types as needed
            pass

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
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
    if os.path.exists("output.txt"):
        with open("output.txt", "r") as f:
            weights = [x for x in f]
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
    
    with open("output.txt", "w") as f:
        f.write("".join(weights))
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
    seed(1)
    # Load and merge data
    folder_path = os.path.join(os.getcwd(), "perceptron\\src\\data")
    dataset = load_csv(folder_path)
    
    for i in range(len(dataset[1])-1):
        str_column_to_float(dataset, i)
    # convert string class to integers
    str_column_to_int(dataset, len(dataset[1])-1)
	# evaluate algorithm
    n_folds = 3
    l_rate = 0.01
    n_epoch = 250
    scores = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

if __name__ == "__main__":
	main()