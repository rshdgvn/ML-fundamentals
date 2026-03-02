from sklearn.svm import SVC
from sklearn.metrics import classification_report

svm_model = SVC()
svm_model = svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
print(classification_report(y_pred, y_test))