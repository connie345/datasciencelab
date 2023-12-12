import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ValidationCurveDisplay
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


#### Setup

# Fetch the MNIST dataset
mnist = fetch_openml(name="mnist_784",version=1,parser='auto')

# Access the data and target labels
X, y = mnist.data.to_numpy(), mnist.target.to_numpy()

le = LabelEncoder()
le.fit(y)
y = le.transform(y)

print(X.shape,y.shape)

# Create split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#### Part 1

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(
    n_estimators=170,
    max_depth=25,
    min_samples_split=4,
    min_samples_leaf=2,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", accuracy)

# Perform 5-fold cross-validation and obtain predicted probabilities
y_pred_proba = cross_val_predict(rf_classifier, X, y, cv=5, method='predict_proba')

# Calculate the log loss
logloss = log_loss(y, y_pred_proba)

# Calculate and print the mean log loss
print(f"The Log Loss: {logloss:.4f}")



#### Part 2

# Create a XGBoost Classifier
xgb_classifier = xgb.XGBClassifier(
    n_estimators=600,       # Number of boosting rounds (trees)
    learning_rate=0.1,      # Step size shrinking to prevent overfitting
    max_depth=6,            # Maximum depth of individual trees
    n_jobs=-1,              
    verbosity=2,
    random_state=42         # Random seed for reproducibility
)

# Train the classifier on the training data
xgb_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = xgb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", accuracy)

# Train the classifier on the training Data
xgb_classifier.fit(X_train, y_train)

# Calculate the log loss
y_pred_proba = xgb_classifier.predict_proba(X_test)
logloss = log_loss(y_test, y_pred_proba)

# Calculate and print the mean log loss
print(f"The Log Loss: {logloss:.4f}")