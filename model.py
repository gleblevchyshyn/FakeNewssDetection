import pandas as pd

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn import metrics


data = pd.read_csv("preparedData.csv")

vec = TfidfVectorizer()

vec.fit(data['title'].values.astype('U'))
s = vec.transform(data['title'].values.astype('U')).toarray()

X_train, X_test, y_train, y_test = train_test_split(s, data['label'], test_size=0.33, random_state=10)

lr_parameters = {'C': [0.1, 1, 10]}
lr = GridSearchCV(LogisticRegression(), lr_parameters, n_jobs=-1, cv=10)

sgd_parameters = {'penalty': ['l2', 'l1'],
                  'alpha': [0.0001, 0.1, 10]}
sgd = GridSearchCV(SGDClassifier(), sgd_parameters, n_jobs=-1, cv=10)

print("Logistic Regression training...")
lr.fit(X_train, y_train)
print("SGB training...")
sgd.fit(X_train, y_train)

print("Logistic Regression best_score:{0}, best_params:{1}, cv_results:{2}".format(lr.best_score_, lr.best_params_, lr.best_estimator_))
print("SGD best_score:{0}, best_params:{1}, cv_results:{2}".format(sgd.best_score_, sgd.best_params_, sgd.best_estimator_))

print("Training models on best params...")
lr = LogisticRegression(**lr.best_params_)
sgd = SGDClassifier(**sgd.best_params_)
lr.fit(X_train, y_train)
sgd.fit(X_train, y_train)



y_lr = lr.predict_proba(X_test)
y_sgd = sgd.predict_proba(X_test)
fpr_lr, tpr_lr, _ = metrics.roc_curve(y_test,  y_lr)
fpr_sgd, tpr_sgd, _ = metrics.roc_curve(y_test,  y_sgd)











