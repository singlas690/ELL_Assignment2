import numpy as np 

# Returns the accuracy of model
# Given two [m x 1] arrays of prediction and actual labels
def model_accuracy(Y_pred, Y):
	return np.sum(Y_pred == Y) / Y_pred.shape[0]

# Returns an array [n x 1] of model recall per class
# Given two [m x 1] arrays of prediction and actual labels and number of classes n
# https://stats.stackexchange.com/questions/51296/how-do-you-calculate-precision-and-recall-for-multiclass-classification-using-co
def model_recall(Y_pred, Y, n):
	classes = (np.arange(n)).reshape((n,1))
	temp_y_pred = np.repeat(Y_pred.T, n, axis = 0)
	temp_y_org = np.repeat(Y.T, n, axis = 0)
	tp = (np.sum(np.logical_and((temp_y_pred == classes), (temp_y_org == classes)), axis = 1)).reshape(n,1)
	tp_fn = (np.sum((temp_y_org == classes), axis = 1)).reshape(n,1)
	return tp/tp_fn

# Returns an array [n x 1] of model precision per class
# Given two [m x 1] arrays of prediction and actual labels and number of classes n
def model_precision(Y_pred, Y, n):
	classes = (np.arange(n)).reshape((n,1))
	temp_y_pred = np.repeat(Y_pred.T, n, axis = 0)
	temp_y_org = np.repeat(Y.T, n, axis = 0)
	tp = (np.sum(np.logical_and((temp_y_pred == classes), (temp_y_org == classes)), axis = 1)).reshape(n,1)
	tp_fp = (np.sum((temp_y_pred == classes), axis = 1)).reshape(n,1)
	return tp/tp_fp

# Returns an array [n x 1] of model f1 score per class
# Given two [m x 1] arrays of prediction and actual labels and number of classes n
def model_f1(Y_pred, Y, n):
	recall = model_recall(Y_pred, Y, n)
	precision = model_precision(Y_pred, Y, n)
	return 2*recall*precision/(recall + precision)

def model_macro_average(Y_pred, Y, n):
	macro_recall = np.sum(model_recall(Y_pred, Y, n))/n
	macro_precision = np.sum(model_precision(Y_pred, Y, n))/n
	return np.asarray([macro_precision, macro_recall, 2*macro_precision*macro_recall/(macro_recall + macro_precision)])

def model_micro_average(Y_pred, Y, n):
	classes = (np.arange(n)).reshape((n,1))
	temp_y_pred = np.repeat(Y_pred.T, n, axis = 0)
	temp_y_org = np.repeat(Y.T, n, axis = 0)
	tp = (np.sum(np.logical_and((temp_y_pred == classes), (temp_y_org == classes)), axis = 1)).reshape(n,1)
	tp_fp = (np.sum((temp_y_pred == classes), axis = 1)).reshape(n,1)
	tp_fn = (np.sum((temp_y_org == classes), axis = 1)).reshape(n,1)
	
	micro_precision = np.sum(tp)/np.sum(tp_fp)
	micro_recall = np.sum(tp)/np.sum(tp_fn)
	
	return np.asarray([micro_precision, micro_recall, 2*micro_precision*micro_recall/(micro_recall + micro_precision)])


if __name__ == "__main__" :
	print("Testing Functions \n")
	a = (3*np.random.rand(10,1)).astype("int")
	b = (3*np.random.rand(10,1)).astype("int")

	print(np.hstack((a,b)))
	print("\nAccuracy = %f \n" % model_accuracy(a,b))
	print("\nPrecision")
	print(model_precision(a,b,3))
	print("\nRecall \n")
	print(model_recall(a,b,3))
	print("\nF1 Score \n")
	print(model_f1(a,b,3))