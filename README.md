# Multivariate-Regression
Obtained a data set suitable for regression, and apply linear regression plus one other regression technique.

This program uses Multivariate Regression dataset from the UCI Machine Learning repository. Here, we have used dataset that was built on the behaviour of the urban traffic of the city of Sao Paula in Brazil and our target is to predict the percentage of traffic created by using 17 different features. We have used this dataset and ran Linear Regressor, KNeighbors Regressor, Decision Tree Regressor and MLP Regressor to compare which algorithm predicts closer to the target.
Source: https://archive.ics.uci.edu/ml/datasets/Behavior+of+the+urban+traffic+of+the+city+of+Sao+Paulo+in+Brazil

After running all the algorithms with the same dataset KNeighbors Regressor outperformed others in 10 different runs. And I believe it outperformed because the dataset scattered all over the place with a lot of outliers and due to that decision tree regressor is not able to classify properly because it is overfitting, MLP Regressor is not able to converge and find the clear distribution. Whereas, KNeighbors Regressor is able to perform well because it is trying to predict target using the target of K-nearest points from the testing point and it is doing well because it is not overfitting and it also takes average of 3 nearest points to predict the target so it has lowest RSS error rate.

![alt text](https://github.com/prerakpatelca/Multivariate-Regression/blob/master/Screen%20Shot%202020-12-28%20at%206.34.58%20PM.png)
