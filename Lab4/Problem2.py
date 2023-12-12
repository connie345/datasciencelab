import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn.model_selection import ValidationCurveDisplay
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score


#### Part 1

# Fetch the MNIST dataset
mnist = fetch_openml(name="mnist_784",version=1,parser='auto')

# Access the data and target labels
X, y = mnist.data.to_numpy(), mnist.target.to_numpy()

print(X.shape,y.shape)

# Check to make sure that we are getting images
def display_mnist_image(image_data, label):
    # Reshape the image data to 28x28 pixels
    image_data = image_data.reshape(28, 28)

    plt.figure(figsize=(4, 4))  # Set the figure size
    plt.imshow(image_data, cmap='gray')  # Display the image in grayscale
    plt.title(f"Label: {label}")  # Display the label as the title
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()

display_mnist_image(X[0],y[0])

#### Part 2

# Create split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a StratifiedKFold cross-validation splitter with a random seed
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create LogisticRegression Model
lg = LogisticRegressionCV(
    multi_class='multinomial',
    penalty='elasticnet',
    cv=cv_splitter,
    Cs=[0.05,0.1,.5,1],
    solver='saga',
    verbose=1,
    n_jobs=-1,
    l1_ratios=[.25,.5,.75],
    max_iter=40,
    random_state=42
)

# Fit the model to the training data
lg.fit(X_train,y_train)

# Predict probabilities on the test and training data
scores_test = lg.predict_proba(X_test)
scores_train = lg.predict_proba(X_train)

# Calculate the log loss
logloss_test = log_loss(y_test, scores_test)
logloss_train = log_loss(y_train, scores_train)

# Display the log loss for test and training data
print("Log Loss (Test Data):", logloss_test)
print("Log Loss (Training Data):", logloss_train)

# Get the best C and l1_ratio values
best_C = lg.C_[0]
best_l1_ratio = lg.l1_ratio_[0]

# Display the best hyperparameters
print("Best C:", best_C)
print("Best l1_ratio:", best_l1_ratio)

##### PART 3


# Define a list of candidate C values to iterate over
C_values = [0.0001,0.001, 0.01, 0.1, 1.0]

# Iterate over each C value and fit a logistic regression model
for C in C_values:
    # Create a LogisticRegression model with the current C value
    model = LogisticRegression(
        C=C,
        multi_class='multinomial',
        penalty='l1',  # Use 'l1' penalty for logistic regression
        solver='saga',  # Use 'saga' solver for logistic regression
        max_iter=40,   # Specify the maximum number of iterations
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    # Fit the model to the data
    model.fit(X_train, y_train)

    # Calculate the log loss
    scores = model.predict_proba(X_test)
    log_loss_value = log_loss(y_test, scores)

    # Calculate sparsity as a percentage
    sparsity_percentage = np.mean(model.coef_ == 0) * 100

    # Print the log loss and sparsity for the current C value
    print(f"For C={C:.4f} - Log Loss: {log_loss_value:.4f}, Sparsity: {sparsity_percentage:.2f}%")

#### Part 4


model = LogisticRegression(
    C=0.01,
    multi_class='multinomial',
    penalty='l1',  # Use 'l1' penalty for logistic regression
    solver='saga',  # Use 'saga' solver for logistic regression
    max_iter=40,   # Specify the maximum number of iterations
    verbose=1,
    n_jobs=-1,
    random_state=42
)
model.fit(X,y)


def plot_coefficients_heatmap(coefficients,classifier_name):
    # Reshape the coefficients into a 28x28 grid
    coefficients = coefficients.reshape(28, 28)

    # Create a heatmap using Seaborn
    plt.figure(figsize=(8, 6))  # Set the figure size
    sns.heatmap(coefficients, annot=False, cmap='coolwarm', cbar=True)

    # Add labels and title
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Index")
    plt.title(f"{classifier_name} Coefficients Heatmap")

    plt.show()

for val in range(10):
    plot_coefficients_heatmap(model.coef_[val],str(val))


#### Extra

# Getting the accuracy of the model

model = LogisticRegression(
    C=0.001,
    multi_class='multinomial',
    penalty='l1',  # Use 'l1' penalty for logistic regression
    solver='saga',  # Use 'saga' solver for logistic regression
    max_iter=40,   # Specify the maximum number of iterations
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# Perform 5-fold cross-validation and obtain predicted probabilities
score = cross_val_score(model, X, y, cv=5)

# Calculate and print the mean log loss
print(f"The Accuracy: {score.mean():.4f}")