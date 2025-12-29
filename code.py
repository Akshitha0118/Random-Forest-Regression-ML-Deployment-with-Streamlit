

# import the libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# read the dataset
dataset=pd.read_csv(r'C:\Users\ADMIN\Downloads\23rd- Poly\23rd- Poly\1.POLYNOMIAL REGRESSION\emp_sal.csv')

# x and y variables
x=dataset.iloc[: , 1:2].values
y = dataset.iloc[:,2].values

# random forest 
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(max_depth=4,criterion= "poisson",random_state=0,n_estimators=6)
rf_reg.fit(x,y)

y_pred_rf =rf_reg.predict([[6.5]])
y_pred_rf
