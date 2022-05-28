#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import os

#%%
# An instance of the Series class is created and initialized with a list of values.
data1 = pd.Series(data= [1.2, 3.4, 4.7, 6.7], name="values")
print(data1) 


# %%
# Iterate on the values of a serie
print('Directly iterate on the values')
for values in data1:
    print(values)
print()
    
# It's also possible to use serie's index for iteration
print("Same but we use the index :")
for i in data1.index:
    print("index : {}, values: {}".format(i, data1[i]))

# %%
# We create an instance of the DataFrame class and initialize it with the values
data_dict = {
    'name': ['Agnes', 'Sidi', 'Thibault', 'Samia', 'Henry', 'Georges'],
    'age': [28, 37, 43, 33, 29, 57],
    'job': ['web analyst', 'sales director', 'web analyst', 'sales director', 
                   'web analyst', 'developer']
            }

data2 = pd.DataFrame(data_dict)
print(data2)  # Equivalent de print() mais avec un meilleur rendu
# %%
# Like Series, DataFrame has an attribut 'index':
print(data2.index)

# %%
# The 'columns' attribute is used to retrieve the list of column names:
print(data2.columns)

# %%
# The shape attribute returns the number of rows and columns as a tuple:
print(data2.shape)

# %%
# The 'values' attribute allows to retrieve the values stored in the DataFrame in numpy.array format:
print(data2.values)

# %%
# See an overview of the first 5 lines of the DataFrame
data2.head()

# %%
# See an overview of the last 5 lines of the DataFrame
data2.tail()

# %%
# Select one column
print(data2['name'])
print()

# %%
# Select multiple columns
liste_col = ['name','job']
print(data2[liste_col])
print()

# %%
# Select sub-part of the DataFrame with slices
# Select the first three lines of the DataFrame
print("three first lines of the DataFrame, with every columns:")
print(data2.loc[0:2,:])
print()

# Select three first line of the 'age' column
print("three first lines of the 'age' column:")
print(data2.loc[0:2,'age'])
print()

# Select the fourth line of 'age' and 'profesion'
print("fourth line of 'age' and 'profession' column:")
print(data2.loc[3,['age', 'job']])


# %%
# Use iloc to access the columns via their position:
print(data2.iloc[:,2])

# With iloc, we can also use negative clues:
print(data2.iloc[:,-1])

#%%
# Use masks to select rows according to a certain condition:
mask = data2['age'] > 30
data2.loc[mask,['age','job']]
# %%
# Add a column whose values are calculated according to another column: apply/lambda functions

# New column containing the square of the age
data2['age_squared'] = data2['age'].apply(lambda x : x**2)

# New column containing: age if age > 30, 0 otherwise
data2['age_changed'] = data2['age'].apply(lambda x : x if x > 30 else 0)

# New column indicating that the person is NOT a web analyst
data2['not_web_analyst'] = data2['job'].apply(lambda x : x != 'web analyst')
# %%
print(data2)

# %%
countries = pd.DataFrame({'Country_Names': ['China', 'United States', 'Japan', 'United Kingdom', 'Russian Federation', 'Brazil'],
                          'Values1': [1.5, 10.53, 7.542, 3.487, 6.565, 8.189],
                          'Values2': [1,2,3,4,5,6]}).set_index('Country_Names')
print(countries.index)

# %%
df_chipotle = pd.read_csv('chipotle.csv')

# %%
type(df_chipotle['item_price'][0])
# %%
df_chipotle.groupby("item_price").order_id.mean()
# %%
