import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('cleaned.csv')

features = np.array(data.drop('Label',axis=1))

labels = np.array(data['Label'])

# # without train-test split
# clf = SVC(kernel = 'linear')
# clf.fit(features,labels)

# print(classification_report(labels,clf.predict(features)))

# # with train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42, stratify=labels)

# # svm
# clf = SVC(kernel = 'linear')
# # tree
# clf = DecisionTreeClassifier()
# forest
clf = RandomForestClassifier(verbose=2,n_jobs=-1)

clf.fit(X_train,y_train)

print(classification_report(y_test,clf.predict(X_test)))

# # cross-validation and hyperparameter tuning
# pip_clf = Pipeline([
#     ('clf', SVC(kernel = 'linear'))
# ])

# parameters = {
#   'clf__kernel': ['linear','poly']
# }

# gs_clf = GridSearchCV(pip_clf, parameters, cv=5, verbose=2, n_jobs=-1)

# gs_clf = gs_clf.fit(X_train, y_train)

# print(classification_report(y_test,gs_clf.predict(X_test)))