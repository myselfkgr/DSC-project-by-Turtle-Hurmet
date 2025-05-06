import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# Load data with correct delimiter
df = pd.read_csv("/content/student-mat.csv", sep=';')

# Lowercase column names
df.columns = df.columns.str.strip().str.lower()

# Manual binary mapping (safe with all known values)
binary_map = {
    'sex': {'M': 0, 'F': 1},
    'address': {'U': 0, 'R': 1},
    'pstatus': {'A': 0, 'T': 1},
    'schoolsup': {'no': 0, 'yes': 1},
    'famsup': {'no': 0, 'yes': 1},
    'paid': {'no': 0, 'yes': 1},
    'activities': {'no': 0, 'yes': 1},
    'nursery': {'no': 0, 'yes': 1},
    'higher': {'no': 0, 'yes': 1},
    'internet': {'no': 0, 'yes': 1},
    'romantic': {'no': 0, 'yes': 1},
    'guardian': {'mother': 0, 'father': 1, 'other': 2}
}

# Apply binary mapping
for col, mapping in binary_map.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# Identify remaining non-numeric columns and encode them
non_numeric_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=non_numeric_cols)

# Drop rows with NaN (e.g., if mapping failed)
df.dropna(inplace=True)

# Select features and target
X = df.drop(columns='g3')
y = df['g3']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Fit KNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("Classification Error:", 1 - acc)
print("Sensitivity (micro):", recall_score(y_test, y_pred, average='micro'))

if len(np.unique(y)) == 2:
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    print("Specificity:", specificity)
else:
    print("Specificity: Not defined for multi-class classification")
