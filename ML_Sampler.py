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
