#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd  
from pandas import Series, DataFrame 
 
import seaborn as sns 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


iris = pd.read_csv("C:/Users/poorvika/Desktop/Iris.csv") 


# In[14]:


iris.head()
   


# In[15]:


iris.info()


# In[16]:


iris.drop("Id", axis=1, inplace = True)


# In[17]:


import matplotlib.pyplot as plt

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(10, 7))

# Plot each species on the same axis
iris[iris.Species == 'Iris-setosa'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', ax=ax, color='blue', label='Iris-setosa')
iris[iris.Species == 'Iris-versicolor'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', ax=ax, color='green', label='Iris-versicolor')
iris[iris.Species == 'Iris-virginica'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', ax=ax, color='red', label='Iris-virginica')

# Set labels and title
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_title('Sepal Length vs Sepal Width')

# Show the legend
ax.legend()

# Display the plot
plt.show()


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a FacetGrid
g = sns.FacetGrid(iris, hue='Species', height=5)

# Map the scatter plot to the FacetGrid
g.map(plt.scatter, 'SepalLengthCm', 'SepalWidthCm')

# Add a legend
g.add_legend()

# Display the plot
plt.show()



# In[24]:


import matplotlib.pyplot as plt

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(10, 7))

# Plot each species on the same axis
iris[iris.Species == 'Iris-setosa'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', ax=ax, color='black', label='Iris-setosa')
iris[iris.Species == 'Iris-versicolor'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', ax=ax, color='blue', label='Iris-versicolor')
iris[iris.Species == 'Iris-virginica'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', ax=ax, color='red', label='Iris-virginica')

# Set labels and title
ax.set_xlabel('Petal Length')
ax.set_ylabel('Petal Width')
ax.set_title('Petal Length vs Petal Width')

# Show the legend
ax.legend()

# Display the plot
plt.show()


# In[23]:


iris.hist(edgecolor='red', linewidth=1.2) 
fig = plt.gcf() 
fig.set_size_inches(12,6) 
plt.show() 


# In[25]:


plt.figure(figsize=(15,10)) 
plt.subplot(2,2,1) 
sns.violinplot(x='Species', y = 'SepalLengthCm', data=iris) 
plt.subplot(2,2,2) 
sns.violinplot(x='Species', y = 'SepalWidthCm', data=iris) 
 
plt.subplot(2,2,3) 
sns.violinplot(x='Species', y = 'PetalLengthCm', data=iris) 
plt.subplot(2,2,4) 
sns.violinplot(x='Species', y = 'PetalWidthCm', data=iris)


# In[27]:


# Importing all the necessary packages to use various classification algorithms
from sklearn.linear_model import LogisticRegression  # For Logistic Regression Algorithm
from sklearn.model_selection import train_test_split  # To split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # KNN classifier
from sklearn import svm  # For Support Vector Machine algorithm
from sklearn import metrics  # For checking the model accuracy
from sklearn.tree import DecisionTreeClassifier  # For Decision Tree Algorithm


# In[28]:


iris.shape


# In[31]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Create a figure with specified size
plt.figure(figsize=(8, 4))

# Draw a heatmap with annotations and a color map
sns.heatmap(iris_df.corr(), annot=True, cmap='cubehelix_r')

# Display the plot
plt.show()




# In[47]:


iris = pd.read_csv("C:/Users/poorvika/Desktop/Iris.csv") 
train, test = train_test_split(iris, test_size=0.3)
print(train.shape) 
print(test.shape) 
iris.head()


# In[48]:


train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] # 
train_y = train.Species # output of the training data 

test_X = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] # ta
test_y = test.Species # output value of the test data 


# In[49]:


train_X.head()


# In[50]:


test_X.head() 


# In[51]:


train_y.head()


# In[52]:


model = svm.SVC() # select the svm algorithm 

# we train the algorithm with training data and training output 
model.fit(train_X, train_y) 

# we pass the testing data to the stored algorithm to predict the outcome 
prediction = model.predict(test_X) 
print('The accuracy of the SVM is: ', metrics.accuracy_score(prediction, test_y)) #
#we pass the predicted output by the model and the actual output 


# In[53]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Create and train the model
model = LogisticRegression()
model.fit(train_X, train_y)

# Make predictions
predictions = model.predict(test_X)

# Calculate and print the accuracy
accuracy = metrics.accuracy_score(test_y, predictions)

print('The accuracy of Logistic Regression is:', accuracy)


# In[56]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Create and train the model
model = DecisionTreeClassifier()
model.fit(train_X, train_y)

# Make predictions
predictions = model.predict(test_X)

# Calculate and print the accuracy
accuracy = metrics.accuracy_score(test_y, predictions)
print('The accuracy of Decision Tree is:', accuracy)


# In[59]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Create and train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(train_X, train_y)

# Make predictions
predictions = model.predict(test_X)

# Calculate and print the accuracy
accuracy = metrics.accuracy_score(test_y, predictions)
print('The accuracy of KNN is:', accuracy)


# In[61]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Create an empty Series with float dtype
a = pd.Series(dtype=float)
a_index = list(range(1, 11))

# Loop through different values of n_neighbors
for i in a_index:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(train_X, train_y)
    predictions = model.predict(test_X)
    accuracy = metrics.accuracy_score(test_y, predictions)
    a = pd.concat([a, pd.Series([accuracy])], ignore_index=True)

# Plot the accuracy values
plt.plot(a_index, a, marker='o')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs. Number of Neighbors')
plt.xticks(a_index)  # Set x-ticks to match the range of neighbors
plt.grid(True)
plt.show()


# In[62]:


petal = iris[['PetalLengthCm','PetalWidthCm','Species']] 
sepal = iris[['SepalLengthCm','SepalWidthCm','Species']]


# In[63]:


train_p,test_p = train_test_split(petal, test_size=0.3, random_state=0) #petals 
train_x_p = train_p[['PetalWidthCm','PetalLengthCm']] 
train_y_p = train_p.Species 
test_x_p = test_p[['PetalWidthCm','PetalLengthCm']] 
test_y_p = test_p.Species 


# In[64]:


train_s,test_s = train_test_split(sepal, test_size=0.3, random_state=0) #sepals 
train_x_s = train_s[['SepalWidthCm','SepalLengthCm']] 
train_y_s = train_s.Species 
test_x_s = test_s[['SepalWidthCm','SepalLengthCm']] 
test_y_s = test_s.Species 


# In[66]:


from sklearn import svm
from sklearn import metrics

# First SVM model using petal features
model_petal = svm.SVC() 
model_petal.fit(train_x_p, train_y_p)  
predictions_petal = model_petal.predict(test_x_p)  
accuracy_petal = metrics.accuracy_score(test_y_p, predictions_petal)
print('The accuracy of the SVM using Petals is:', accuracy_petal)

# Second SVM model using sepal features
model_sepal = svm.SVC() 
model_sepal.fit(train_x_s, train_y_s)  
predictions_sepal = model_sepal.predict(test_x_s)  
accuracy_sepal = metrics.accuracy_score(test_y_s, predictions_sepal)
print('The accuracy of the SVM using Sepals is:', accuracy_sepal)


# In[10]:


# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (Sepal Length, Sepal Width, Petal Length, Petal Width)
y = iris.target  # Labels (0, 1, 2 for the three Iris species)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling (optional but recommended for logistic regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=200)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of the Logistic Regression model:', accuracy)

# Detailed classification report
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=iris.target_names))


# In[2]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()

# Features (Petals and Sepals)
X = iris.data
# This contains both petal and sepal measurements
y = iris.target
# The target labels (flower species)

# Split the dataset into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree model
model = DecisionTreeClassifier()

# Train the model using the training data (petals and sepals combined)
model.fit(train_x, train_y)

# Predict the labels for the test set
predictions = model.predict(test_x)

# Calculate and print the accuracy
accuracy = accuracy_score(test_y, predictions)
print('The accuracy of the Decision Tree Classifier is:', accuracy)

# Visualize the Decision Tree
plt.figure(figsize=(12,8))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()


# In[1]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Create the KNeighborsClassifier model
model = KNeighborsClassifier(n_neighbors=3)

# Fit the model using Petal features
model.fit(train_x_p, train_y_p)
# Predict using the test Petal features
prediction_p = model.predict(test_x_p)
# Print the accuracy for Petal features
print('The accuracy of the KNN using Petals is:', metrics.accuracy_score(test_y_p, prediction_p))

# Fit the model using Sepal features
model.fit(train_x_s, train_y_s)
# Predict using the test Sepal features
prediction_s = model.predict(test_x_s)
# Print the accuracy for Sepal features
print('The accuracy of the KNN using Sepals is:', metrics.accuracy_score(test_y_s, prediction_s))


# In[4]:


import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# Define markers and colors
markers = ('s', 'x', 'o')
colors = ('red', 'blue', 'lightgreen')
cmap = ListedColormap(colors[:len(np.unique(y_test))])

# Plot data points
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                c=cmap(idx), marker=markers[idx], label=cl)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='best')
plt.show()


# In[5]:


import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# Load your dataset
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

# Define markers and colors
markers = ('s', 'x', 'o')
colors = ('red', 'blue', 'lightgreen')
cmap = ListedColormap(colors[:len(np.unique(y))])

# Plot data points
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], c=cmap(idx), marker=markers[idx], label=iris.target_names[cl])

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='best')
plt.show()




# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot all data points
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    
    # highlight test samples
    if test_idx is not None:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='none',
                    edgecolor='black', alpha=1.0, linewidth=1, marker='o',
                    s=55, label="test set")
    
    plt.legend(loc='best')
    plt.show()


# In[8]:


from sklearn.svm import SVC

# Initialize the SVM classifier with an RBF kernel
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=1.0)

# Fit the model on the training data
svm.fit(X_train_std, y_train)

# Print the accuracy of the SVM classifier on the training data
print('The accuracy of the SVM classifier on training data is {:.2f} out of 1'.format(svm.score(X_train_std, y_train)))

# Print the accuracy of the SVM classifier on the test data
print('The accuracy of the SVM classifier on test data is {:.2f} out of 1'.format(svm.score(X_test_std, y_test)))


# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import models from Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Import model evaluation tools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve

# Import data
from sklearn.datasets import load_breast_cancer


# In[ ]:




