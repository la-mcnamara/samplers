#########################################################
#    Title: Code Sampler                                #
#   Author: Lauren McNamara                             #
#  Created: 4/11/2019                                   #                       
# Modified: 4/11/2019                                   #
#  Purpose: Quick reference for some ML techniques      #
#########################################################


##############  Setup  ##############
# import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np 
import scipy.stats as stats
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# set working directory
os.chdir("/Users/Lauren/Documents/Python/Iris")
os.getcwd()
os.listdir('.')

# color palette
# source: https://learnui.design/tools/data-color-picker.html#palette
pdblue = '#003f5c'
plblue = '#444e86'
ppurple = '#955196'
ppink = '#dd5182'
porange = '#ff6e54'
pyellow = '#ffa600'
pgray = '#64666B'


##############  Get Data  ##############
# Fisher's iris data
iris = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/d546eaee765268bf2f487608c537c05e22e4b221/iris.csv')
iris.head()

setosa = iris[iris['species']=='setosa']
versicolor = iris[iris['species']=='versicolor']
virginica = iris[iris['species']=='virginica']


##############  Reconfigure Data  ##############

# add dummy variables for species
dummies = pd.get_dummies(iris.species)
dummies.head()
iris = iris.join(dummies)


##############  Visualize Data  ##############
# bee swarm plots
def beeswarm(var, label):
    sns.swarmplot(x='species', y=var, data=iris) 
    plt.xlabel('Species')
    plt.ylabel(label)
    plt.show()

beeswarm('sepal_length', 'Sepal Length')
beeswarm('sepal_width', 'Sepal Width')
beeswarm('petal_length', 'Petal Length')
beeswarm('petal_width', 'Petal Width')


##############  Linear Regression  ##############
# predict sepal length
X=iris[['setosa','versicolor','sepal_width','petal_length','petal_width']] # dropping virginica as reference group
y= iris['sepal_length']

####  Using sklearn
lm=linear_model.LinearRegression()
model = lm.fit(X,y)

predictions = lm.predict(X)
predictions[0:5]

# R-squared
lm.score(X,y)

# intercept
lm.intercept_

# coefficients
lm.coef_

####  Using statsmodels
X = sm.add_constant(X) # add intercept (beta_0) to model
model = sm.OLS(y, X).fit()
predictions = model.predict(X)
predictions[0:5]

# output model stats
model.summary()


##############  Logistic Regression with feature selection  ##############
X=iris[['sepal_length','sepal_width','petal_length','petal_width']] # dropping species
y= iris['virginica'] # predict vriginica

# use sklearn rfe to select top 3 features
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, y)

# output
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))
# sepal width is not selected
