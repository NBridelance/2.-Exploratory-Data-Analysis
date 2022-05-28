#https://app.jedha.co/course/data-manipulation-exercises-pt/explore-a-db-pt
# Exploring a database
# Let's practice on Pandas. Download the dataset chipotle.csv from JULIE and upload it to your workspace.


#%%
import pandas as pd

#%%
df_chipotle = pd.read_csv('chipotle.csv')

#%%
df_chipotle.head(10)

# %%
df_chipotle.shape

#%%
# %%
df_chipotle.columns


#%%
# Most ordered item
df_chipotle.groupby('item_name').quantity.sum().sort_values(ascending = False)
#%%
# Sum of ordered item
df_chipotle.quantity.sum()

#%%
df_chipotle['item_price_rounded'] = df_chipotle.item_price.str.replace('$','').astype(float)
# %%
df_chipotle.head(10)

#%%
df_chipotle['total_price'] = df_chipotle['quantity'] * df_chipotle['item_price_rounded']

#%%
# Total sales
print('The turnover of this dataset is $'+str(df_chipotle['total_price'].sum()))
# %%
print('The average revenue per order is $'+str(round(df_chipotle['total_price'].mean(),2)))