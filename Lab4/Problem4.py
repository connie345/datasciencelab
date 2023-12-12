import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import ValidationCurveDisplay
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score


#### Setup

# Fetch the CIFAR-10-Small dataset
cifar_10_small = fetch_openml(name="CIFAR_10_small", version=1,parser='auto')

# Access the data and target labels
X, y = cifar_10_small.data, cifar_10_small.target

print(X.shape,y.shape)

# Create split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#### Part 1

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(
    n_estimators=400,
    max_depth=25,
    min_samples_split=4,
    min_samples_leaf=2,
    n_jobs=-1,
    verbose=1,
    random_state=42
)
### Training

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", accuracy)

### Accuracy Cross Validation

# Perform 5-fold cross-validation and calculate accuracies
accuracies = cross_val_score(rf_classifier, X, y, cv=5, n_jobs=-1)

# Print the accuracy for each fold
for fold, accuracy in enumerate(accuracies, 1):
    print(f"Fold {fold}: Accuracy = {accuracy:.4f}")

# Calculate and print the mean accuracy
mean_accuracy = np.mean(accuracies)
print(f"Mean Accuracy = {mean_accuracy:.4f}")


### Loss Cross validation

# Perform 5-fold cross-validation and obtain predicted probabilities
y_pred_proba = cross_val_predict(rf_classifier, X, y, cv=5, method='predict_proba')

# Calculate the log loss
logloss = log_loss(y, y_pred_proba)

# Calculate and print the mean log loss
print(f"The Log Loss: {logloss:.4f}")


#### Part 2

# Create a LightGBM classifier
lgb_classifier = lgb.LGBMClassifier(
    n_estimators=650,      # Number of boosting rounds (trees)
    learning_rate=0.1,     # Step size shrinking to prevent overfitting
    max_depth=6,           # Maximum depth of individual trees
    n_jobs=-1,             # Use all available CPU cores
    verbosity=1,           # Verbosity level
    random_state=42        # Random seed for reproducibility
)

# Train the classifier on the training data
lgb_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lgb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", accuracy)
