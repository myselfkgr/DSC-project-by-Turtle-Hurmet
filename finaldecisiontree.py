import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

# Load dataset
df = pd.read_csv('student-mat.csv', delimiter=';')

# Convert categorical variables to numeric
binary_mappings = {
    'sex': {'M': 0, 'F': 1},
    'school': {'GP': 0, 'MS': 1},
    'address': {'U': 0, 'R': 1},
    'famsize': {'LE3': 0, 'GT3': 1},
    'Pstatus': {'T': 0, 'A': 1},
    'schoolsup': {'yes': 1, 'no': 0},
    'famsup': {'yes': 1, 'no': 0},
    'paid': {'yes': 1, 'no': 0},
    'activities': {'yes': 1, 'no': 0},
    'nursery': {'yes': 1, 'no': 0},
    'higher': {'yes': 1, 'no': 0},
    'internet': {'yes': 1, 'no': 0},
    'romantic': {'yes': 1, 'no': 0},
    'guardian': {'mother': 0, 'father': 1, 'other': 2}
}

for col, mapping in binary_mappings.items():
    df[col] = df[col].map(mapping)

# Define features and target
predictors = df.iloc[:, :8].values
target = df.iloc[:, 32].values  # G3 column (final grade)

# Train-test split
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, test_size=0.25, random_state=42)

# Train Decision Tree
classifier = DecisionTreeClassifier(criterion='entropy', random_state=1, splitter='best')
classifier.fit(pred_train, tar_train)

# Predictions
predictions = classifier.predict(pred_test)

# Evaluation
print("Accuracy of training dataset is: {:.2f}".format(classifier.score(pred_train, tar_train)))
print("Accuracy of test dataset is: {:.2f}".format(classifier.score(pred_test, tar_test)))
print("Error rate is:", 1 - accuracy_score(tar_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(tar_test, predictions))

# Sensitivity (Recall for positive class)
print("Sensitivity is:", recall_score(tar_test, predictions, average='micro'))
# Specificity (1 - recall for negative class – simplified)
print("Specificity is:", 1 - recall_score(tar_test, predictions, average='micro'))

# Plot the decision tree
plt.figure(figsize=(20,10))
features = list(df.columns[:8])
plot_tree(classifier, feature_names=features, class_names=None, filled=True, rounded=True)
plt.show()
