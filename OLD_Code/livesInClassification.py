import numpy as np
from sklearn import linear_model,neighbors
from sklearn.ensemble import RandomForestClassifier
import math
from sklearn.preprocessing import scale
from sklearn import svm, datasets
import pickle
from sklearn.metrics import precision_recall_fscore_support

def calculatePRF(yt, yp, l=""):
	s = "Printing precision, recall, and f-score for "
	print s + l
	print "Macro: ",
	print precision_recall_fscore_support(yt, yp, average='binary')
	# print "Micro: ",
	# print precision_recall_fscore_support(yt, yp, average='binary')

def logisticRegression(filename_train, filename_test):
	data = pickle.load(open(filename_train))
	data_X,data_Y= data['X'],data['Y']
	dataTest = pickle.load(open(filename_test))
	np.concatenate((data_X,dataTest['X']),axis=0)
	np.concatenate((data_Y,dataTest['Y']),axis=0)

	lenOfdata = np.shape(data_X)[0]
	print "Length of total data: " + str(lenOfdata)
	trainingSize = (int) (lenOfdata * 0.80)
	print "Length of training data: " + str(trainingSize)
	X_train = data_X[0:trainingSize]
	X_test = data_X[trainingSize:]

	# Split the targets into training/testing sets
	Y_train = data_Y[0:trainingSize]
	Y_test = data_Y[trainingSize:]

	# Create linear regression object
	knn = neighbors.KNeighborsClassifier()
	logistic = linear_model.LogisticRegression()
	rbf_svc = svm.SVC(kernel='linear')
	rfc = RandomForestClassifier(n_estimators=10)
	# mlp = MLPClassifier(alpha=0.1, learning_rate_init=0.1)
	knn_fit = knn.fit(X_train, Y_train)
	logistic_fit = logistic.fit(X_train, Y_train)
	svm_fit = rbf_svc.fit(X_train, Y_train)
	rfc_fit = rfc.fit(X_train, Y_train)

	print('KNN score: %f' % knn_fit.score(X_test, Y_test))
	print('LogisticRegression score: %f' % logistic_fit.score(X_test, Y_test))
	print('SVM (rbf) score: %f' % svm_fit.score(X_test, Y_test))
	print('Random Forest Classifier score: %f' % rfc_fit.score(X_test, Y_test))
	# print('MLP score: %f' % mlp.fit(X_train, Y_train).score(X_test, Y_test))

	calculatePRF(Y_test, knn_fit.predict(X_test), "KNN")
	calculatePRF(Y_test, logistic_fit.predict(X_test), "Logistic Regression")
	calculatePRF(Y_test, svm_fit.predict(X_test), "SVM(rbf)")
	calculatePRF(Y_test, rfc_fit.predict(X_test), "Random Forest Classifier")

def rfMany(filename_train, filename_test):
	data = pickle.load(open(filename_train))
	data_X,data_Y= data['X'],data['Y']
	dataTest = pickle.load(open(filename_test))
	np.concatenate((data_X,dataTest['X']),axis=0)
	np.concatenate((data_Y,dataTest['Y']),axis=0)

	lenOfdata = np.shape(data_X)[0]
	# print "Length of total data: " + str(lenOfdata)
	trainingSize = (int) (lenOfdata * 0.80)
	# print "Length of training data: " + str(trainingSize)
	X_train = data_X[0:trainingSize]
	X_test = data_X[trainingSize:]

	# Split the targets into training/testing sets
	Y_train = data_Y[0:trainingSize]
	Y_test = data_Y[trainingSize:]
	maxi = 0
	max_tuple = ()
	max_ne = 0
	print "Testing many RFs..."
	for ne in range(1,11):
		print "RF with number of estimators:",ne
		rfc = RandomForestClassifier(n_estimators=ne)
		for i in range(1):
			rfc_fit = rfc.fit(X_train, Y_train)
			temp = precision_recall_fscore_support(Y_test, rfc_fit.predict(X_test), average='binary')
			if temp[2]>maxi:
				maxi = temp[2]
				max_tuple = temp
				max_ne = ne
	print max_tuple
	print "Max with:",max_ne

def svmMany(filename_train, filename_test):
	print "\n\n"
	data = pickle.load(open(filename_train))
	data_X,data_Y= data['X'],data['Y']
	dataTest = pickle.load(open(filename_test))
	np.concatenate((data_X,dataTest['X']),axis=0)
	np.concatenate((data_Y,dataTest['Y']),axis=0)

	lenOfdata = np.shape(data_X)[0]
	# print "Length of total data: " + str(lenOfdata)
	trainingSize = (int) (lenOfdata * 0.80)
	# print "Length of training data: " + str(trainingSize)
	X_train = data_X[0:trainingSize]
	X_test = data_X[trainingSize:]

	# Split the targets into training/testing sets
	Y_train = data_Y[0:trainingSize]
	Y_test = data_Y[trainingSize:]

	kernels = ['linear','rbf','poly','sigmoid']
	Cparams = [1.0,0.5,1.5,0.1,2.0,2.5,3.0,3.5,4.0,4.5]
	for k in kernels:
		for c in Cparams:
			print k,c			
			rbf_svc = svm.SVC(C=c,kernel=k)
			svm_fit = rbf_svc.fit(X_train, Y_train)
			calculatePRF(Y_test, svm_fit.predict(X_test), "SVM")

if __name__ == "__main__":
	logisticRegression('./trainedFiles/feature_train.pkl','./trainedFiles/feature_dev.pkl')
	rfMany('./trainedFiles/feature_train.pkl','./trainedFiles/feature_dev.pkl')
	svmMany('./trainedFiles/feature_train.pkl','./trainedFiles/feature_dev.pkl')