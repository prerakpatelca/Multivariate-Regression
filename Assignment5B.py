""" This program uses Multivariate Regression dataset from the UCI Machine Learning repository. Here, we have used dataset that was built on the behavior of the urban traffic of the city of Sao Paula in Brazil and our target is to predict the precentage of traffic created by using 17 different features. We have used this dataset and ran Linear Regressor, KNeighbors Regressor, Decision Tree Regressor and MLP Regressor to compare which alogrithm predicts closer to the target.
Source: https://archive.ics.uci.edu/ml/datasets/Behavior+of+the+urban+traffic+of+the+city+of+Sao+Paulo+in+Brazil

After running all the algorithms with the same dataset KNeighbors Regressor outperformed others in 10 different runs. And I believe it outperformed because the dataset scattered all over the place with a lot of outliers and due to that decision tree regressor is not able to classify properly because it is overfitting, MLP Regressor is not able to converge and find the clear distribution. Whereas, KNeighbors Regressor is able to perform well because it is trying to predict target using the target of K-nearest points from the testing point and it is doing well because it is not overfitting and it also takes average of 3 nearest points to predict the target so it has lowest RSS error rate.

Prerak Patel, Student, Mohawk College, 2020
"""

# importing libraries
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np

## Load the traffic data
trafficdata=[]
traffictargets=[]
with open("traffic.csv") as file:
    for line in file:
        row = line.strip().split(",")
        eachrow = []
        for item in row:
            eachrow += [float(item)]
        trafficdata+=[eachrow[:-1]]
        traffictargets+=[eachrow[-1]]

# declaring the training data as a numpy array
trafficdata = np.array(trafficdata)
traffictargets = np.array(traffictargets)

# shuffling the data and getting indexes
indexes = np.random.permutation(len(trafficdata))
# declaring a split of 75 - 25
split = round(len(trafficdata)*0.25)

## splitting traffic data into data and targets
# using 75 split for the training data
trainingdata=trafficdata[indexes[split:]]
trainingtargets=traffictargets[indexes[split:]]
# using 25 split for the testing data
testingdata=trafficdata[indexes[:split]]
testingtargets=traffictargets[indexes[:split]]


# Output the size of the training and testing sets and the number of features.
print("Size of Training set = " + str(trainingdata.shape[0]) + "\n" +
"Size of Testing set = " + str(testingdata.shape[0]) + "\n" +
"Number of features = " + str(trainingdata.shape[1]) + "\n")

## Declaring and using LinearRegressor
print("LinearRegressor Results")
linearregression = LinearRegression()
# passing training data and training targets to linear regressor
linearregression.fit(trainingdata, trainingtargets)
# passing testing data to linear regressor to predict the data
testpred = linearregression.predict(testingdata)

# Output the RSS error (e), correlation (r), weights, and intercept for a linear regression.
print("Weights: " + str(np.around(linearregression.coef_,decimals=3)) + "\n" +
"Intercept: " + str(linearregression.intercept_) + "\n" +
"Correlation: " + str(np.corrcoef(testpred,testingtargets)[0,1]) + "\n" +
"RSS error: " + str(((testpred-testingtargets)**2).sum()) + "\n")

## Declaring and using KNeighborsRegressor
print("KNeighborsRegressor Results")
# initializing the KNeighborsRegressor with best parameters
neigh = KNeighborsRegressor(n_neighbors=3,leaf_size=20,p=1)
# passing training data and training targets to train
neigh.fit(trainingdata, trainingtargets)
# passing testing data and testing targets to predict the targets
neightestingpred = neigh.predict(testingdata)

# Output correlation and Residual Sum of Squares for KNeighborsRegressor
print("Correlation: " + str(np.corrcoef(neightestingpred,testingtargets)[0,1]) + "\n" +
"RSS error: " + str(((neightestingpred-testingtargets)**2).sum()) + "\n")

## Declaring and using DecisionTreeRegressor
print("Decision Tree Results")
# initializing the DecisionTreeRegressor with best parameters
decisiontree = DecisionTreeRegressor(max_depth=5)
# passing training data and training targets to decision tree
decisiontree.fit(trainingdata, trainingtargets)
# passing testing data and testing targets to decision tree
decisiontreetestingpred = decisiontree.predict(testingdata)

# Output correlation and Residual Sum of Squares for DecisionTreeRegressor
print("Correlation: " + str(np.corrcoef(decisiontreetestingpred,testingtargets)[0,1]) + "\n" +
"RSS error: " + str(((decisiontreetestingpred-testingtargets)**2).sum()) + "\n")

## Declaring and using MLP Regressor
print("MLP Regressor Results")
# initializing the MLPRegressor with best parameters
mlpregressor = MLPRegressor(max_iter=10000, learning_rate = 'adaptive',solver='lbfgs')
# passing training data and training targets to MLP Regressor
mlpregressor.fit(trainingdata, trainingtargets)
# passing testing data and testing targets to MLP Regressor
mlpregressortestingpred = mlpregressor.predict(testingdata)

# Output correlation and Residual Sum of Squares for MLPRegressor
print("Correlation: " + str(np.corrcoef(mlpregressortestingpred,testingtargets)[0,1]) + "\n" +
"RSS error: " + str(((mlpregressortestingpred-testingtargets)**2).sum()) + "\n")











