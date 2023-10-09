from random import seed
from random import randrange
from csv import reader
import pandas as pd

def load_and_merge_csv(filename):
    dataset = pd.read_csv(filename)

    customer_info = pd.read_csv('customers.csv')
    merchant_info = pd.read_csv('merchants.csv')
    terminal_info = pd.read_csv('terminals.csv')

    dataset = dataset.merge(customer_info, on='CUSTOMER_ID', how='left')
    dataset = dataset.merge(merchant_info, on='MERCHANT_ID', how='left')
    dataset = dataset.merge(terminal_info, on='TERMINAL_ID', how='left')

    dataset = dataset.values.tolist()

    return dataset

def preprocess_dataset(dataset): 
	# Convert 'IS_RECURRING_TRANSACTION' to a boolean
    dataset['IS_RECURRING_TRANSACTION'] = dataset['IS_RECURRING_TRANSACTION'].astype(bool)

    # Convert 'TX_AMOUNT' to float
    dataset['TX_AMOUNT'] = dataset['TX_AMOUNT'].astype(float)

    # Convert 'TX_FRAUD' to int
    dataset['TX_FRAUD'] = dataset['TX_FRAUD'].astype(int)

    # Convert 'TRANSACTION_GOODS_AND_SERVICES_AMOUNT' to float
    dataset['TRANSACTION_GOODS_AND_SERVICES_AMOUNT'] = dataset['TRANSACTION_GOODS_AND_SERVICES_AMOUNT'].astype(float)

    # Convert 'TRANSACTION_CASHBACK_AMOUNT' to float
    dataset['TRANSACTION_CASHBACK_AMOUNT'] = dataset['TRANSACTION_CASHBACK_AMOUNT'].astype(float)

    return dataset

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
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for _ in range(len(train[0]))]
	for _ in range(n_epoch):
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
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
    filename = 'transaction_train.csv'
    dataset = load_and_merge_csv(filename)
    
    # Preprocess the dataset
    dataset = preprocess_dataset(dataset)
    
    # evaluate algorithm
    n_folds = 3
    l_rate = 0.01
    n_epoch = 500
    scores = evaluate_algorithm(dataset.values.tolist(), perceptron, n_folds, l_rate, n_epoch)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))