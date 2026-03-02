from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

lg_model = LogisticRegression()
lg_model = lg_model.fit(X_train, y_train)

y_pred = lg_model.predict(X_test)
print(classification_report(y_pred, y_test))