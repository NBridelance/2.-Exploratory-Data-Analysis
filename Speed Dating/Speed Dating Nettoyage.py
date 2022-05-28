#%%
# Imports
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA


#%%
# Set current directory to /home/utilisateur/Python/DVF
os.chdir('/home/utilisateur/Python/FullStack/Exploratory Data Analysis/Speed Dating')
print(os.getcwd())


#%%
# Import & visualize dataset
df = pd.read_csv("Speed Dating Cleaned.csv")
df.head()

#%%
df = df.drop('id', axis=1)
#%%
df = df.drop('field_cd', axis=1)
#%%
df = df.drop('idg', axis=1)
#%%
df = df.drop('career_c', axis=1)


#%%
df.gender = df.gender.replace(0,'F')
df.gender = df.gender.replace(1,'M')
df.head()

#%%
df.match = df.match.replace('N', 0)
df.match = df.match.replace('Y', 1)

#%%
df['match'].head()

#%%
df.dec_o = df.dec_o.replace('N', 0)
df.dec_o = df.dec_o.replace('Y', 1)
df['match'].head()

#%%
df.dec = df.dec.replace('N', 0)
df.dec = df.dec.replace('Y', 1)
df.head()

#%%
df.date_3 = df.date_3.map({0:'N',1:'Y'})
df.head()

#%%
df.race = df.race.replace(1,'African American')
df.race = df.race.replace(2,'Caucasian')
df.race = df.race.replace(3,'Hispanic')
df.race = df.race.replace(4,'Asian')
df.race = df.race.replace(5,'Native')
df.race = df.race.replace(6,'Other')
df.head()

#%%
df.goal = df.goal.replace(1,'Fun')
df.goal = df.goal.replace(2,'Meet people')
df.goal = df.goal.replace(3,'Get a date')
df.goal = df.goal.replace(4,'Relationship')
df.goal = df.goal.replace(5,'Say I did it')
df.goal = df.goal.replace(6,'Other')
df.head()

#%%
# Export dataset to CSV
df.to_csv(r'/home/utilisateur/Python/FullStack/Exploratory Data Analysis/Speed Dating/Speed Dating Cleaned.csv', index = False)