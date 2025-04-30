import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import sklearn.metrics

df = pd.read_csv('/Users/kgr/Downloads/student+performance/student/student-mat.csv')
df['sex'] = df['sex'].map({'M': 0, 'F': 1})
df['address'] = df['address'].map({'U': 0, 'R': 1})
df['guardian'] = df['guardian'].map({'mother': 0, 'father': 1})
#df['internet'] = df['internet'].map({'no': 0, 'yes': 1})


predictor_columns = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
                     'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
                     'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                     'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout',
                     'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']

predictors = df[predictor_columns].values
targets = df["G3"].values

pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size= 0.25)


print(pred_train.shape)
print(pred_test.shape)
print(tar_train.shape)
print(tar_test.shape)

neigh = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto')
neigh.fit(pred_train, tar_train)
y_pred = neigh.predict(pred_test)


#accuracy
print("Accuracy is ", accuracy_score(tar_test, y_pred, normalize = True))
#classification error
print("Classification error is",1- accuracy_score(tar_test, y_pred, normalize = True))
#sensitivity
print("sensitivity is", sklearn.metrics.recall_score(tar_test, y_pred, labels=None, average =  'micro', sample_weight=None))
#specificity
print("specificity is", 1 - sklearn.metrics.recall_score(tar_test, y_pred,labels=None, average =  'micro', sample_weight=None))
