# House Market
# https://app.jedha.co/course/data-manipulation-exercises-pt/house-market-pt


#%%
import pandas as pd

#%%
df_house_price = pd.read_csv('house_price.csv')
df_nb_bathrooms = pd.read_csv('number_of_bathrooms.csv')
df_nb_rooms = pd.read_csv('number_of_rooms.csv')
df_square_ft = pd.read_csv('square_feet.csv')

#%%
df_square_ft.columns = ['id', 'surface']
df_nb_bathrooms.columns = ['id','bathrooms']
df_nb_rooms.columns = ['id', 'rooms'] 
# %%
df_house_price.head()
# %%
df_nb_bathrooms.head()
# %%
df_nb_rooms.head()
# %%
df_square_ft.head() 
#%%
df_merged = df_house_price.merge(df_nb_rooms, on='id', how='inner')
#%%
df_merged = df_merged.merge(df_nb_bathrooms, on='id', how='inner')
#%%
df_merged = df_merged.merge(df_square_ft, on='id', how='inner')

#%%
df_merged = df_square_ft.merge(df_nb_bathrooms, on='id').merge(df_nb_rooms, on='id').merge(df_house_price, on='id')
# see documentation : merge performs an inner join by default dataset.head() 
#%%
df_merged.to_csv('house_merged.csv')
#%%
print('The average surface area in this dataset is '+str(df_merged.nb_sq_ft.mean())+' square feet')

#%%
print('The average number of room is '+str(df_merged.nb_of_rooms.mean())+', and the median number is '+str(df_merged.nb_of_rooms.median()))
# %%
# What is the average cost of a house, depending on the number of rooms it has?
df_merged.groupby('nb_of_rooms').house_price.mean().round(2)

# %%
# Create a new column in your dataset that we'll call home_size Create three categories that respectively correspond to :
# "very large" == "a house larger than 25,000 sqrt_feet"
# "large" == "a house between 20,000 and 25,000 sqrt_feet"
# "medium" == "a house between 15,000 and 20,000 sqrt_feet"
# "small" == "a house between 10,000 and 15,000 sqrt_feet"
# "very small" == a house less than 10,000 sqrt_feet"

# %%
df_merged['house_size_type'] = df_merged['nb_sq_ft'].apply(lambda x :   'very large' if x > 25000 else
                                                                        'large' if x > 20000 else
                                                                        'medium' if x > 15000 else
                                                                        'small' if x > 10000 else
                                                                        'tiny')
# %%
df_merged.groupby('house_size_type').house_price.mean().round(2).sort_values(ascending=False)

#%%
df_merged[df_merged['house_size_type']=='tiny']

#%%
df_merged.plot.scatter(x='nb_sq_ft', y='house_price')

#%%
import seaborn as sns
sns.lmplot( x = "nb_sq_ft",# abscissa
            y = "house_price", # ordinate 
            data = df_merged, # wich datas 
            line_kws={'color':'red'}); # line color 
# %%
