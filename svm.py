import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import label_binarize
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# Load the data for Mathematics (assuming it's stored in a CSV file)
math_data = pd.read_csv('student-mat.csv', delimiter=';')

# Convert categorical variables to numerical using one-hot encoding
math_data = pd.get_dummies(math_data, columns=['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
                                               'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'])

selected_features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime',
                     'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']  # Features

X_mat = math_data[selected_features]  # Feature matrix X_mat
y_mat = math_data['G3']

# Splitting the data into training and testing sets for Mathematics dataset
X_train_mat, X_test_mat, y_train_mat, y_test_mat = train_test_split(X_mat, y_mat, test_size=0.2, random_state=42)

# Create an SVM classifier for Mathematics using One-vs-One strategy
svm_classifier_mat = OneVsOneClassifier(SVC(kernel='linear'))

# Train the SVM classifier for Mathematics
svm_classifier_mat.fit(X_train_mat, y_train_mat)

# Make predictions on Mathematics test set
predictions_mat = svm_classifier_mat.predict(X_test_mat)

# Calculate accuracy for Mathematics predictions
accuracy_mat = accuracy_score(y_test_mat, predictions_mat)
print(f"Accuracy for Mathematics dataset: {accuracy_mat}")

# Filter out classes with no positive samples
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UndefinedMetricWarning)
    y_test_bin = label_binarize(y_test_mat, classes=np.unique(y_mat))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(np.unique(y_mat))):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], svm_classifier_mat.decision_function(X_test_mat)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), svm_classifier_mat.decision_function(X_test_mat).ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure(figsize=(8, 6))
plt.plot(fpr["micro"], tpr["micro"], color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Mathematics dataset')
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix and Classification Report
conf_mat = confusion_matrix(y_test_mat, predictions_mat)
print("Confusion Matrix:\n", conf_mat)
print("\nClassification Report:\n", classification_report(y_test_mat, predictions_mat))