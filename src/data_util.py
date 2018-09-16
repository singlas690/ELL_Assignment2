import numpy as np
import pandas as pd

# Reference - https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
# Returns flattened images and correponding labels
def load_fashion_mnist(path, label_present = True):
	"""Load MNIST data from `path`"""
	labels_path = path + '_labels/data'
	images_path = path + '_images/data'
	
	labels = []
	if label_present:
		with open(labels_path, 'rb') as lbpath:
			labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)	
		labels = np.expand_dims(labels, axis = 1)

	with open(images_path, 'rb') as imgpath:
		images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

	if label_present:
		return images, labels

	return images

def load_medical(path, label_map, label_present = True):
	df = pd.read_csv(path)
	df.fillna(df.mean(), inplace=True)

	df = df.applymap(lambda s: label_map.get(s) if s in label_map else s)

	X = df[['TEST1' , 'TEST2', 'TEST3']].as_matrix()
	
	if label_present:
		Y = df[['Health']].as_matrix()
		return X, Y

	return X

def load_railway(path, sex_map, class_map, label_present = True, one_hot = True):
	df = pd.read_csv(path)
	df.fillna(df.mode().iloc[0], inplace=True)

	df = df.applymap(lambda s: sex_map.get(s) if s in sex_map else s)
	df = df.applymap(lambda s: class_map.get(s) if s in class_map else s)

	X = df[['budget', 'memberCount', 'preferredClass', 'sex', 'age']].as_matrix()

	unique_pc = np.unique((df['preferredClass'].values))
	pC = np.zeros(X.shape[0], unique_pc[-1])
	pC[np.arange(X.shape[0]), df['preferredClass'].values] = 1

	sx = np.zeros(X.shape[0], 1)
	sx[np.arange(X.shape[0]), df['sex'].values] = 1

	X_one_hot = np.hstack((df[['budget', 'memberCount', 'age']].as_matrix(), pC, sx)) 

	if label_present:
		Y = df[['boarded']].as_matrix()
		if one_hot:
			return X_one_hot, Y
		return X, Y

	if one_hot:
		return X_one_hot
	return X 

def split_data(X, Y, train_ratio, random_state = 0):
	data = np.random.RandomState(seed = random_state).permutation(np.hstack((X, Y)))

	data_train = data[:int(train_ratio * data.shape[0]),:]
	data_test = data[int(train_ratio * data.shape[0]):,:]

	return data_train[:, :-1], data_train[:, -1:], data_test[:, :-1], data_test[:, -1:]

def normalize(X, std_tol = 0.00001):
	X = X.astype('float')
	mu = np.mean(X, axis = 0)
	std = np.std(X, axis = 0)
	std[std < std_tol] = 1

	return (X - mu)/std, mu, std


def normalize2(X, std_tol = 0.00001):
	X = X.astype('float')
	
	norm = np.expand_dims(np.linalg.norm(X, axis = 1) , axis = 1)
	norm[norm < std_tol] = 1

	return X/norm
