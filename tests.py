from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from classifiers import CoTrainingClassifier


if __name__ == '__main__':
	N_SAMPLES = 25000
	N_FEATURES = 1000
	X, y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES)

	y[:N_SAMPLES//2] = -1

	X_test = X[-N_SAMPLES//4:]
	y_test = y[-N_SAMPLES//4:]

	X_labeled = X[N_SAMPLES//2:-N_SAMPLES//4]
	y_labeled = y[N_SAMPLES//2:-N_SAMPLES//4]

	y = y[:-N_SAMPLES//4]
	X = X[:-N_SAMPLES//4]


	X1 = X[:,:N_FEATURES // 2]
	X2 = X[:, N_FEATURES // 2:]




	print 'Logistic'
	base_lr = LogisticRegression()
	base_lr.fit(X_labeled, y_labeled)
	y_pred = base_lr.predict(X_test)
	print classification_report(y_test, y_pred)

	print 'Logistic CoTraining'
	lg_co_clf = CoTrainingClassifier(LogisticRegression())
	lg_co_clf.fit(X1, X2, y)
	y_pred = lg_co_clf.predict(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
	print classification_report(y_test, y_pred)

	print 'SVM'
	base_svm = LinearSVC()
	base_svm.fit(X_labeled, y_labeled)
	y_pred = base_lr.predict(X_test)
	print classification_report(y_test, y_pred)
	
	print 'SVM CoTraining'
	svm_co_clf = CoTrainingClassifier(LinearSVC(), u=N_SAMPLES//10)
	svm_co_clf.fit(X1, X2, y)
	y_pred = svm_co_clf.predict(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
	print classification_report(y_test, y_pred)
	
	
