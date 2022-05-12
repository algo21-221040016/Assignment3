# Assignment3

In this assignment, I tried to use the machine learning method
to participate in a data-analysis competition.

#Note:
The detailed data can not be uploaded to github due to the request
of the organization committee.

#Details of the model
I choosed to use XGboost and lightGBM as well as catboost in this 
model. Meanwhile, I tried model stacking in the project. 

These two .py are models with the highest auc score, where the stacking
model is used in the first test set and the featuretools and 
lightGBM is used in the second test set.

Actually, data in the first test set has almost the same distribution
with the training set, thus the auc score is very high. And stacking 
then can have better performance.

However, the data in the second test set has a very different distribution
with the training set. As a result, feature selection is very important.
It's very difficult to choose better features and I tried RFE and adversarial
validation as well as the kde of features this time. Meanwhile, I use featuretools
to construct new features.
But finally, I still 
didn't get the best features. I guess that I should try to consider
this question from a commercial perspective.

Anyway, the boosting is a very useful machine learning method, but
in many cases, feature engineering can decide the upper limit of the 
results.