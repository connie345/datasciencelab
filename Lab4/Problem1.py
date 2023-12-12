import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ValidationCurveDisplay
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

##### Part 1

# Fetch the CIFAR-10-Small dataset
cifar_10_small = fetch_openml(name="CIFAR_10_small", version=1,parser='auto')

# Access the data and target labels
X, y = cifar_10_small.data, cifar_10_small.target

print(X.shape,y.shape)


#### Part 2

# Display one of the images

# Helper Function to Display the image
def display_image(image_data):
    image = image_data.reshape(3, 32, 32).transpose(1, 2, 0)

    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Selecting which index to show
X = X.to_numpy()

display_image(X[1]) # Cool truck

##### Part 3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

##### Part 4

# Going forward I will be using cross_val_score() for cross validation.

lg = LogisticRegression(multi_class='multinomial',penalty='elasticnet',C=1,solver='saga', verbose=1, n_jobs=-1, l1_ratio=.5,random_state=42)
scores = cross_val_score(lg,X,y,cv=4)
print(scores.mean())

# Create Model
lg = LogisticRegression(multi_class='multinomial',penalty='elasticnet',C=1,solver='saga', verbose=1, n_jobs=-1, l1_ratio=.5,max_iter=100,random_state=42)
lg.fit(X_train,y_train)

# Predict probabilities for training and test data
train_probabilities = lg.predict_proba(X_train)
test_probabilities = lg.predict_proba(X_test)

# Calculate log loss for training and test data
train_log_loss = log_loss(y_train, train_probabilities)
test_log_loss = log_loss(y_test, test_probabilities)

print(train_log_loss,test_log_loss)


lg = LogisticRegression(multi_class='multinomial',penalty='l1',C=0.001,solver='saga', verbose=1, n_jobs=-1,max_iter=1,warm_start=True,random_state=42)
# Calculate the number of columns to eliminate (40% of total columns)
num_columns_to_eliminate = int(0.85 * 3072)

# Generate random column indices to eliminate
random_indices = np.random.choice(3072, size=num_columns_to_eliminate, replace=False)
epochs = 50
training_loss = []
validation_loss = []
for epoch in range(epochs):
    lg = lg.fit(X_train, y_train) 
    # lg.coef_[:, :30] = 0
    # lg.coef_[:, -30:] = 0
    # Set the columns specified by random indices to zero
    lg.coef_[:, random_indices] = 0
    Y_pred = lg.predict(X_train)
    curr_train_score = accuracy_score(y_train, Y_pred)   # training performances
    Y_pred = lg.predict(X_test) 
    curr_valid_score = accuracy_score(y_test, Y_pred)    # validation performances
    training_loss.append(curr_train_score)               # list of training perf to plot
    validation_loss.append(curr_valid_score)             # list of valid perf to plot


plt.plot(range(epochs),training_loss,label='Train')
plt.plot(range(epochs),validation_loss,label='Test')
plt.legend()
plt.show()

coef_lg = lg.coef_.ravel()
sparsity_lg = np.mean(coef_lg == 0) * 100
print("{:<40} {:.2f}%".format("Sparsity with L1 penalty:", sparsity_lg))
