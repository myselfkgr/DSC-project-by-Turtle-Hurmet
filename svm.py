import time
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsOneClassifier
from sklearn.exceptions import UndefinedMetricWarning

# Load dataset
math_data = pd.read_csv('student-mat.csv', delimiter=';')

# Convert categorical variables using one-hot encoding
math_data = pd.get_dummies(math_data, columns=[
    'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
    'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
    'nursery', 'higher', 'internet', 'romantic'
])

# Selected features
selected_features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime',
                     'failures', 'famrel', 'freetime', 'goout', 'Dalc',
                     'Walc', 'health', 'absences', 'G1', 'G2']

X_mat = math_data[selected_features]  # Feature matrix
y_mat = math_data['G3']               # Target

# Split data
X_train_mat, X_test_mat, y_train_mat, y_test_mat = train_test_split(
    X_mat, y_mat, test_size=0.2, random_state=42
)

# Train One-vs-One SVM Classifier
svm_classifier_mat = OneVsOneClassifier(SVC(kernel='linear', probability=True))
svm_classifier_mat.fit(X_train_mat, y_train_mat)

# Make predictions
predictions_mat = svm_classifier_mat.predict(X_test_mat)

# Evaluate performance
accuracy_mat = accuracy_score(y_test_mat, predictions_mat)
print("Accuracy for Mathematics dataset:", accuracy_mat)

# Confusion Matrix and Classification Report
conf_mat = confusion_matrix(y_test_mat, predictions_mat)
print("Confusion Matrix:\n", conf_mat)
print("\nClassification Report:\n", classification_report(y_test_mat, predictions_mat))

# ROC Curve and AUC (micro-average)
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

# Micro-average ROC
fpr["micro"], tpr["micro"], _ = roc_curve(
    y_test_bin.ravel(), svm_classifier_mat.decision_function(X_test_mat).ravel()
)
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
         lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Mathematics dataset')
plt.legend(loc="lower right")
plt.show()
