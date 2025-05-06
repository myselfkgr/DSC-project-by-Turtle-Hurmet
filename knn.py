# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score
import sklearn.metrics

# Load the dataset
df = pd.read_csv("/content/student-mat.csv", delimiter=';')

# Convert categorical variables to numerical
df['sex'] = df['sex'].map({'M': 0, 'F': 1})
df['address'] = df['address'].map({'U': 0, 'R': 1})
df['guardian'] = df['guardian'].map({'mother': 0, 'father': 1, 'other':2}) #add 'other' to handle all categories
# Add more mappings if necessary for other categorical columns...

# Define predictor and target columns
# Instead of removing 'G3', define predictor columns as all columns except 'G3'
predictor_columns = df.columns.tolist()
predictor_columns.remove('G3') # Remove the target column

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['school', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
                                 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher',
                                 'internet', 'romantic'])

#After one-hot encoding, update predictor_columns to include new columns
predictor_columns = df.columns.tolist()
predictor_columns.remove('G3') #Remove target column again


predictors = df[predictor_columns].values
targets = df['G3'].values  # or use classification e.g., pass/fail

# Split the data into training and testing sets
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=0.25)

# Initialize and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto')
knn.fit(pred_train, tar_train)

# Make predictions
y_pred = knn.predict(pred_test)

# Evaluation
print("Accuracy is:", accuracy_score(tar_test, y_pred, normalize=True))
print("Classification error is:", 1 - accuracy_score(tar_test, y_pred, normalize=True))
print("Sensitivity is:", sklearn.metrics.recall_score(tar_test, y_pred, average='micro'))
print("Specificity is:", 1 - sklearn.metrics.recall_score(tar_test, y_pred, average='micro'))
