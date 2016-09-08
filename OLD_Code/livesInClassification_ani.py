import numpy as np
from sklearn import linear_model,neighbors
from sklearn.ensemble import RandomForestClassifier
import math
from sklearn.preprocessing import scale
from sklearn import svm, datasets
import pickle
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cluster import KMeans


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
	Y_train = data_Y[0:trainingSize]
	Y_test = data_Y[trainingSize:]

	kmeans = KMeans(n_clusters=2)
	kmeans.fit(X_train)
	l = kmeans.predict(X_test)
	ind,ind2 = [],[]
	for i in range(len(l)):
		if l[i]==1:
			ind.append(i)
		else :
			ind2.append(i) 
	xte0,yte0 = X_test[ind2],Y_test[ind2]
	xte1,yte1 = X_test[ind],Y_test[ind]
	l = kmeans.predict(X_train)
	ind,ind2 = [],[]
	for i in range(len(l)):
		if l[i]==1:
			ind.append(i)
		else :
			ind2.append(i) 
	xtr0,ytr0 = X_train[ind2],Y_train[ind2]
	xtr1,ytr1 = X_train[ind],Y_train[ind]
		
	# Split the targets into training/testing sets

	# Create linear regression object
	knn = neighbors.KNeighborsClassifier()
	logistic = linear_model.LogisticRegression()
	rbf_svc = svm.SVC(kernel='linear')
	rfc = RandomForestClassifier(n_estimators=10)
	# mlp = MLPClassifier(alpha=0.1, learning_rate_init=0.1)
	knn_fit = knn.fit(xtr0, ytr0)
	logistic_fit = logistic.fit(xtr0, ytr0)
	svm_fit = rbf_svc.fit(xtr0, ytr0)
	rfc_fit = rfc.fit(xtr0, ytr0)

	print('KNN score: %f' % knn_fit.score(xte0, yte0))
	print('LogisticRegression score: %f' % logistic_fit.score(xte0, yte0))
	print('SVM (rbf) score: %f' % svm_fit.score(xte0, yte0))
	print('Random Forest Classifier score: %f' % rfc_fit.score(xte0, yte0))
	# print('MLP score: %f' % mlp.fit(X_train, Y_train).score(X_test, Y_test))

	calculatePRF(yte0, knn_fit.predict(xte0), "KNN")
	calculatePRF(yte0, logistic_fit.predict(xte0), "Logistic Regression")
	calculatePRF(yte0, svm_fit.predict(xte0), "SVM(rbf)")
	calculatePRF(yte0, rfc_fit.predict(xte0), "Random Forest Classifier")

	knn = neighbors.KNeighborsClassifier()
	logistic = linear_model.LogisticRegression()
	rbf_svc = svm.SVC(kernel='linear')
	rfc = RandomForestClassifier(n_estimators=10)
	# mlp = MLPClassifier(alpha=0.1, learning_rate_init=0.1)
	knn_fit = knn.fit(xtr1, ytr1)
	logistic_fit = logistic.fit(xtr1, ytr1)
	svm_fit = rbf_svc.fit(xtr1, ytr1)
	rfc_fit = rfc.fit(xtr1, ytr1)

	print('KNN score: %f' % knn_fit.score(xte1, yte1))
	print('LogisticRegression score: %f' % logistic_fit.score(xte1, yte1))
	print('SVM (rbf) score: %f' % svm_fit.score(xte1, yte1))
	print('Random Forest Classifier score: %f' % rfc_fit.score(xte1, yte1))
	
	calculatePRF(yte1, knn_fit.predict(xte1), "KNN")
	calculatePRF(yte1, logistic_fit.predict(xte1), "Logistic Regression")
	calculatePRF(yte1, svm_fit.predict(xte1), "SVM(rbf)")
	calculatePRF(yte1, rfc_fit.predict(xte1), "Random Forest Classifier")
	# print('MLP score: %f' % mlp.fit(X_train, Y_train).score(X_test, Y_test))
	return ytr0,ytr1,yte0,yte1,xtr0
	

if __name__ == "__main__":
	ytr0,ytr1,yte0,yte1,xtr0 = logisticRegression('./trainedFiles/feature_train.pkl','./trainedFiles/feature_dev.pkl')
