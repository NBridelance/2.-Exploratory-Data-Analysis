#%%

# Import
import os
from math import sqrt
from gpg import Data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "notebook_connected"
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

#%%

# Set current directory to /home/utilisateur/Python/DVF
os.chdir("/home/utilisateur/Python/FullStack/Exploratory Data Analysis/Speed Dating")
print(os.getcwd())


# Import & visualize dataset
df = pd.read_csv("Speed Dating Cleaned.csv")
print('Dataset loaded !')

#%%
df_individuals = df.drop_duplicates(subset = 'iid')[[   'iid',
                                                        'gender',
                                                        'age', 
                                                        'wave',
                                                        'field', 
                                                        'race',
                                                        'imprace',
                                                        'imprelig',
                                                        'goal',
                                                        'date',
                                                        'go_out',
                                                        'career',
                                                        'sports',
                                                        'tvsports',
                                                        'exercise',
                                                        'dining',
                                                        'museums',
                                                        'art',
                                                        'hiking',
                                                        'gaming',
                                                        'clubbing',
                                                        'reading',
                                                        'tv',
                                                        'theater',
                                                        'movies',
                                                        'concerts',
                                                        'music',
                                                        'shopping',
                                                        'yoga',
                                                        'expnum',
                                                        'attr1_1',
                                                        'sinc1_1',
                                                        'intel1_1',
                                                        'fun1_1',
                                                        'amb1_1',
                                                        'attr4_1',
                                                        'sinc4_1',
                                                        'intel4_1',
                                                        'fun4_1',
                                                        'amb4_1',
                                                        'shar4_1',
                                                        'attr2_1',
                                                        'sinc2_1',
                                                        'intel2_1',
                                                        'fun2_1',
                                                        'amb2_1',
                                                        'shar2_1',
                                                        'attr3_1',
                                                        'sinc3_1',
                                                        'intel3_1',
                                                        'fun3_1',
                                                        'amb3_1',
                                                        'attr5_1',
                                                        'sinc5_1',
                                                        'intel5_1',
                                                        'fun5_1',
                                                        'amb5_1',
                                                        'attr1_s',
                                                        'sinc1_s',
                                                        'intel1_s',
                                                        'fun1_s',
                                                        'amb1_s',
                                                        'attr3_s',
                                                        'sinc3_s',
                                                        'intel3_s',
                                                        'fun3_s',
                                                        'amb3_s'
                                                        ]]

df_individuals = df_individuals.merge(df.groupby('iid')['attr_o'].mean(), on='iid')
df_individuals = df_individuals.merge(df.groupby('iid')['sinc_o'].mean(), on='iid')
df_individuals = df_individuals.merge(df.groupby('iid')['intel_o'].mean(), on='iid')
df_individuals = df_individuals.merge(df.groupby('iid')['fun_o'].mean(), on='iid')
df_individuals = df_individuals.merge(df.groupby('iid')['amb_o'].mean(), on='iid')
df_individuals = df_individuals.merge(df.groupby('iid')['age_o'].mean(), on='iid')
df_individuals = df_individuals.merge(df.groupby('iid')['like_o'].mean(), on='iid')
df_individuals = df_individuals.merge(df.groupby('iid')['match'].sum(), on='iid')
df_individuals = df_individuals.merge(df.groupby('iid')['dec_o'].sum(), on='iid')

df_individuals = df_individuals.merge(df.groupby('iid')['attr'].mean(), on='iid')
df_individuals = df_individuals.merge(df.groupby('iid')['sinc'].mean(), on='iid')
df_individuals = df_individuals.merge(df.groupby('iid')['intel'].mean(), on='iid')
df_individuals = df_individuals.merge(df.groupby('iid')['fun'].mean(), on='iid')
df_individuals = df_individuals.merge(df.groupby('iid')['amb'].mean(), on='iid')
df_individuals = df_individuals.merge(df.groupby('iid')['like'].mean(), on='iid')
df_individuals = df_individuals.merge(df.groupby('iid')['match_es'].mean(), on='iid')
df_individuals = df_individuals.merge(df.groupby('iid')['dec'].sum(), on='iid')

df_individuals = df_individuals.merge(df.groupby('iid')['you_call'].mean(), on='iid')

df_individuals = df_individuals.merge(df.groupby('iid')['them_cal'].mean(), on='iid')

# df_individuals.drop(columns='condtn', inplace=True, axis = 1)


#%%
df_individuals.to_csv('df_individuals.csv')

#%%
df_women = df_individuals[df_individuals['gender']=='F']

df_men = df_individuals[df_individuals['gender']=='M']

#%%
nanglobal = df_individuals.isnull().sum().to_frame()
nanglobal.reset_index(inplace=True)
nanglobal.columns = ['variable','nanglobalcount']
nanmen = df_men.isnull().sum().to_frame()
nanmen.reset_index(inplace=True)
nanmen.columns = ['variable','nanmencount']
nanwomen = df_women.isnull().sum().to_frame()
nanwomen.reset_index(inplace=True)
nanwomen.columns = ['variable','nanwomencount']


                        
#%%
nan = pd.merge(nanglobal, nanmen, on='variable')
nan = pd.merge(nan, nanwomen, on='variable')
nan = nan.sort_values(by='nanglobalcount', ascending=False)

'''
Lots of missing values in
attrx_s : groups were forgotten halfway through (half values missing !)

'''

#%%
df_individuals = df_individuals.sort_values(by='wave')


#%%
px.box(df_women[['sports',
                                                        'tvsports',
                                                        'exercise',
                                                        'dining',
                                                        'museums',
                                                        'art',
                                                        'hiking',
                                                        'gaming',
                                                        'clubbing',
                                                        'reading',
                                                        'tv',
                                                        'theater',
                                                        'movies',
                                                        'concerts',
                                                        'music',
                                                        'shopping',
                                                        'yoga']])


#%%
px.box(df_men[['sports',
                                                        'tvsports',
                                                        'exercise',
                                                        'dining',
                                                        'museums',
                                                        'art',
                                                        'hiking',
                                                        'gaming',
                                                        'clubbing',
                                                        'reading',
                                                        'tv',
                                                        'theater',
                                                        'movies',
                                                        'concerts',
                                                        'music',
                                                        'shopping',
                                                        'yoga']])

#%%


x0 = df_women['match_es']
x1 = df_women['match']

fig = go.Figure()
fig.add_trace(go.Histogram(x=x0))
fig.add_trace(go.Histogram(x=x1))

# Overlay both histograms
fig.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.5)
fig.show()

#%%
px.histogram(df_women[['match_es','match']], barmode="overlay", title='Match prediction of women')
#%%
px.box(df_women[['match_es','match']],  title='Match prediction of women')



#%%
px.histogram(df_men[['match_es','match']], barmode="overlay", title='Match prediction of men')
#%%
px.box(df_men[['match_es','match']],  title='Match prediction of men')

#%%
px.box(df_men[['dec']], title='Number of decisions YES for men')

#%%
px.box(df_women[['dec']], title='Number of decisions YES for women')

#%%
'''
VIZ

attr1_1 what you look for
attr4_1 what you think people look for
attr2_1 what the other sex looks for according to you
attr3_1 how do you see yourself
attr5_1 how do you think the others see you
attr how you judge your partner
attr1_s what you look for, halfway through
attr3_s how do you see yourself, halfway through
'''

#%%
'''
Do people feel unique
'''

categories = ['Attractivity','Sincerity','Intelligence',
           'Fun', 'Ambition']

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
     r=[ df_individuals['attr1_1'].mean(),
     df_individuals['sinc1_1'].mean(),
     df_individuals['intel1_1'].mean(),
     df_individuals['fun1_1'].mean(),
     df_individuals['amb1_1'].mean()],
      theta=categories,
      fill='toself',
      name='What you look for'
))

fig.add_trace(go.Scatterpolar(
     r=[ df_individuals['attr4_1'].mean(),
     df_individuals['sinc4_1'].mean(),
     df_individuals['intel4_1'].mean(),
     df_individuals['fun4_1'].mean(),
     df_individuals['amb4_1'].mean()],
      theta=categories,
      fill='toself',
      name='What you THINK people look for'
))


fig.update_layout(
    title={
        'text': "What people look for vs what they THINK people look for",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    paper_bgcolor = 'white'
)
fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      #range=[5, 9]
    )),
  showlegend=True
)

fig.show()

#%%
px.histogram(df_individuals[['attr']])

#%%
px.histogram(df_individuals[['attr1_1']])

#%%
df.loc[(df['dec'] == 1) & (df['gender'] == 'M'), ['attr']].mean()


#%%
'''
When they said YES
'''

categories = ['Attractivity','Sincerity','Intelligence',
           'Fun', 'Ambition']

fig = go.Figure()


fig.add_trace(go.Scatterpolar(
     r=[ df.loc[(df['dec'] == 1) & (df['gender'] == 'F'), 'attr'].mean(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'F'), 'sinc'].mean(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'F'), 'intel'].mean(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'F'), 'fun'].mean(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'F'), 'amb'].mean()],
      theta=categories,
      fill='toself',
      name='Men said yes to'
))


fig.add_trace(go.Scatterpolar(
     r=[ df.loc[(df['dec'] == 1) & (df['gender'] == 'M'), 'attr'].mean(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'M'), 'sinc'].mean(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'M'), 'intel'].mean(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'M'), 'fun'].mean(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'M'), 'amb'].mean()],
      theta=categories,
      fill='toself',
      name='Women said yes to'
))


fig.update_layout(
    title={
        'text': "When they said YES (M) vs (F)",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    paper_bgcolor = 'white'
)
fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[6, 8]
    )),
  showlegend=True
)

fig.show()

#%%
'''
YES vs NO (M)
'''

categories = ['Attractivity','Sincerity','Intelligence',
           'Fun', 'Ambition']

fig = go.Figure()


fig.add_trace(go.Scatterpolar(
     r=[ df.loc[(df['dec'] == 1) & (df['gender'] == 'F'), 'attr'].mean(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'F'), 'sinc'].mean(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'F'), 'intel'].mean(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'F'), 'fun'].mean(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'F'), 'amb'].mean()],
      theta=categories,
      fill='toself',
      name='Men said yes to'
))


fig.add_trace(go.Scatterpolar(
     r=[ df.loc[(df['dec'] == 0) & (df['gender'] == 'F'), 'attr'].mean(),
     df.loc[(df['dec'] == 0) & (df['gender'] == 'F'), 'sinc'].mean(),
     df.loc[(df['dec'] == 0) & (df['gender'] == 'F'), 'intel'].mean(),
     df.loc[(df['dec'] == 0) & (df['gender'] == 'F'), 'fun'].mean(),
     df.loc[(df['dec'] == 0) & (df['gender'] == 'F'), 'amb'].mean()],
      theta=categories,
      fill='toself',
      name='Men said no to'
))



fig.update_layout(
    title={
        'text': "When women said YES vs when women said NO",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    paper_bgcolor = 'pink'
)
fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[4, 8]
    )),
  showlegend=True
)

fig.show()

#%%
'''
YES vs NO (F)
'''

categories = ['Attractivity','Sincerity','Intelligence',
           'Fun', 'Ambition']

fig = go.Figure()


fig.add_trace(go.Scatterpolar(
     r=[ df.loc[(df['dec'] == 1) & (df['gender'] == 'M'), 'attr'].mean(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'M'), 'sinc'].mean(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'M'), 'intel'].mean(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'M'), 'fun'].mean(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'M'), 'amb'].mean()],
      theta=categories,
      fill='toself',
      name='Women said yes to'
))


fig.add_trace(go.Scatterpolar(
     r=[ df.loc[(df['dec'] == 0) & (df['gender'] == 'M'), 'attr'].mean(),
     df.loc[(df['dec'] == 0) & (df['gender'] == 'M'), 'sinc'].mean(),
     df.loc[(df['dec'] == 0) & (df['gender'] == 'M'), 'intel'].mean(),
     df.loc[(df['dec'] == 0) & (df['gender'] == 'M'), 'fun'].mean(),
     df.loc[(df['dec'] == 0) & (df['gender'] == 'M'), 'amb'].mean()],
      theta=categories,
      fill='toself',
      name='Women said no to'
))



fig.update_layout(
    title={
        'text': "When men said YES vs when they said NO",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    paper_bgcolor = 'lightblue'
)
fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[4, 8]
    )),
  showlegend=True
)

fig.show()

#%%
'''
YES vs NO (both)
'''

categories = ['Attractivity','Sincerity','Intelligence',
           'Fun', 'Ambition']

fig = go.Figure()


fig.add_trace(go.Scatterpolar(
     r=[ df.loc[(df['dec'] == 1) & (df['gender'] == 'M'), 'attr'].median(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'M'), 'sinc'].median(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'M'), 'intel'].median(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'M'), 'fun'].median(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'M'), 'amb'].median()],
      theta=categories,
      fill='toself',
      name='Women said yes to'
))


fig.add_trace(go.Scatterpolar(
     r=[ df.loc[(df['dec'] == 0) & (df['gender'] == 'M'), 'attr'].median(),
     df.loc[(df['dec'] == 0) & (df['gender'] == 'M'), 'sinc'].median(),
     df.loc[(df['dec'] == 0) & (df['gender'] == 'M'), 'intel'].median(),
     df.loc[(df['dec'] == 0) & (df['gender'] == 'M'), 'fun'].median(),
     df.loc[(df['dec'] == 0) & (df['gender'] == 'M'), 'amb'].median()],
      theta=categories,
      fill='toself',
      name='Women said no to'
))

fig.add_trace(go.Scatterpolar(
     r=[ df.loc[(df['dec'] == 1) & (df['gender'] == 'F'), 'attr'].median(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'F'), 'sinc'].median(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'F'), 'intel'].median(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'F'), 'fun'].median(),
     df.loc[(df['dec'] == 1) & (df['gender'] == 'F'), 'amb'].median()],
      theta=categories,
      fill='toself',
      name='Men said yes to'
))


fig.add_trace(go.Scatterpolar(
     r=[ df.loc[(df['dec'] == 0) & (df['gender'] == 'F'), 'attr'].median(),
     df.loc[(df['dec'] == 0) & (df['gender'] == 'F'), 'sinc'].median(),
     df.loc[(df['dec'] == 0) & (df['gender'] == 'F'), 'intel'].median(),
     df.loc[(df['dec'] == 0) & (df['gender'] == 'F'), 'fun'].median(),
     df.loc[(df['dec'] == 0) & (df['gender'] == 'F'), 'amb'].median()],
      theta=categories,
      fill='toself',
      name='Men said no to'
))


fig.update_layout(
    title={
        'text': "YES vs NO : both genders",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    paper_bgcolor = 'white'
)
fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[4, 8]
    )),
  showlegend=True
)

fig.show()

#%%
corre = df.loc[df['dec'] == 1, ['attr','sinc','intel','fun','amb']] #,'attr1_1','sinc1_1','intel1_1','fun1_1','amb1_1']]

corre_corr = corre.corr()

sns.heatmap(corre_corr, xticklabels=corre_corr.columns, yticklabels=corre_corr.columns, cmap='PiYG')


#%%
corre = df.loc[:, ['attr','sinc','intel','fun','amb']] #,'attr1_1','sinc1_1','intel1_1','fun1_1','amb1_1']]

corre_corr = corre.corr()

sns.heatmap(corre_corr, xticklabels=corre_corr.columns, yticklabels=corre_corr.columns, cmap='PiYG')
#%%
px.imshow(corre_corr, text_auto=True, mask=mask_ut)

#%%
mask_ut=np.triu(np.ones(corre_corr.shape)).astype(np.bool)
sns.set(style='white')
sns.heatmap(corre_corr, mask=mask_ut, cmap="Greens")


#%%
corre = df.loc[:, ['attr_o','sinc_o','intel_o','fun_o','amb_o','dec_o']] #,'attr1_1','sinc1_1','intel1_1','fun1_1','amb1_1']]

corre_corr = corre.corr()

sns.heatmap(corre_corr, xticklabels=corre_corr.columns, yticklabels=corre_corr.columns, cmap='PiYG')
#%%
px.imshow(corre_corr, text_auto=True, mask=mask_ut)

#%%
mask_ut=np.triu(np.ones(corre_corr.shape)).astype(np.bool)
sns.set(style='white')
sns.heatmap(corre_corr, mask=mask_ut, cmap="Greens")



#%%
'''
How women see themselves vs. how they judge men
'''

categories = ['Attractivity','Sincerity','Intelligence',
           'Fun', 'Ambition']

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
     r=[ (df_women['attr3_s'].mean() + df_women['attr3_1'].mean())/2,
        (df_women['sinc3_s'].mean() + df_women['sinc3_1'].mean())/2,
        (df_women['intel3_s'].mean() + df_women['intel3_1'].mean())/2,
        (df_women['fun3_s'].mean() + df_women['fun3_1'].mean())/2,
        (df_women['amb3_s'].mean() + df_women['amb3_1'].mean())/2],
      theta=categories,
      fill='toself',
      name='Self-perception'
))

fig.add_trace(go.Scatterpolar(
     r=[ df_women['attr'].mean(),
        df_women['sinc'].mean(),
        df_women['intel'].mean(),
        df_women['fun'].mean(),
        df_women['amb'].mean()],
      theta=categories,
      fill='toself',
      name='Opinion on dated men'
))


fig.update_layout(
    title={
        'text': "How women perceive themselves (blue)<br> vs. <br>how women judge the men they date (red)",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    paper_bgcolor = 'pink'
)
fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[5, 9]
    )),
  showlegend=True
)

fig.show()

#%%
'''
How men see themselves vs. how they judge women
'''

categories = ['Attractivity','Sincerity','Intelligence',
           'Fun', 'Ambition']

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
     r=[ (df_men['attr3_s'].mean() + df_men['attr3_1'].mean())/2,
        (df_men['sinc3_s'].mean() + df_men['sinc3_1'].mean())/2,
        (df_men['intel3_s'].mean() + df_men['intel3_1'].mean())/2,
        (df_men['fun3_s'].mean() + df_men['fun3_1'].mean())/2,
        (df_men['amb3_s'].mean() + df_men['amb3_1'].mean())/2],
      theta=categories,
      fill='toself',
      name='Self-perception'
))

fig.add_trace(go.Scatterpolar(
     r=[ df_men['attr'].mean(),
        df_men['sinc'].mean(),
        df_men['intel'].mean(),
        df_men['fun'].mean(),
        df_men['amb'].mean()],
      theta=categories,
      fill='toself',
      name='Opinion on dated women'
))


fig.update_layout(
    title={
        'text': "How men perceive themselves (blue)<br> vs. <br>how men judge the women they date (red)",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    paper_bgcolor = 'lightblue'
)
fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[5, 9]
    )),
  showlegend=True
)

fig.show()

#%%
'''
How women perceive themselves initially (red) and halfway through (blue)
'''

categories = ['Attractivity','Sincerity','Intelligence',
           'Fun', 'Ambition']

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
     r=[ df_women['attr3_1'].mean(),
        df_women['sinc3_1'].mean(),
        df_women['intel3_1'].mean(),
        df_women['fun3_1'].mean(),
        df_women['amb3_1'].mean()],
      theta=categories,
      fill='toself',
      name='Initial'
))

fig.add_trace(go.Scatterpolar(
     r=[ df_women['attr3_s'].mean(),
        df_women['sinc3_s'].mean(),
        df_women['intel3_s'].mean(),
        df_women['fun3_s'].mean(),
        df_women['amb3_s'].mean()],
      theta=categories,
      fill='toself',
      name='Halfway through'
))


fig.update_layout(
    title={
        'text': "How women perceive themselves initially (red)<br> vs. <br>how women perceive themselves halfway through (blue)",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    paper_bgcolor = 'pink'
)
fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[6.5, 8.5]
    )),
  showlegend=True
)

fig.show()

#%%
'''
How men perceive themselves initially (red) and halfway through (blue)
'''

categories = ['Attractivity','Sincerity','Intelligence',
           'Fun', 'Ambition']

fig = go.Figure()
fig.add_trace(go.Scatterpolar(
     r=[ df_men['attr3_1'].mean(),
        df_men['sinc3_1'].mean(),
        df_men['intel3_1'].mean(),
        df_men['fun3_1'].mean(),
        df_men['amb3_1'].mean()],
      theta=categories,
      fill='toself',
      name='Initial'
))
fig.add_trace(go.Scatterpolar(
     r=[ df_men['attr3_s'].mean(),
        df_men['sinc3_s'].mean(),
        df_men['intel3_s'].mean(),
        df_men['fun3_s'].mean(),
        df_men['amb3_s'].mean()],
      theta=categories,
      fill='toself',
      name='Halfway through'
))

fig.update_layout(
    title={
        'text': "How men perceive themselves initially (red)<br> vs. <br>how men perceive themselves halfway through (blue)",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    paper_bgcolor = 'lightblue'
)
fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[6.5, 8.5]
    )),
  showlegend=True
)

fig.show()


'''
NOTE :
Women tend to feel noticeably MORE attractive after a few dates
Both sexes tend to feel LESS intelligent than initially LESS sincere than initially after a few dates
'''

#%%
'''
How men perceive themselves vs. how women perceive themselves, on average
'''

categories = ['Attractivity','Sincerity','Intelligence',
           'Fun', 'Ambition']

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
     r=[ (df_men['attr3_s'].mean() + df_men['attr3_1'].mean())/2,
        (df_men['sinc3_s'].mean() + df_men['sinc3_1'].mean())/2,
        (df_men['intel3_s'].mean() + df_men['intel3_1'].mean())/2,
        (df_men['fun3_s'].mean() + df_men['fun3_1'].mean())/2,
        (df_men['amb3_s'].mean() + df_men['amb3_1'].mean())/2],
      theta=categories,
      fill='toself',
      name='Men'
))
fig.add_trace(go.Scatterpolar(
     r=[ (df_women['attr3_s'].mean() + df_women['attr3_1'].mean())/2,
        (df_women['sinc3_s'].mean() + df_women['sinc3_1'].mean())/2,
        (df_women['intel3_s'].mean() + df_women['intel3_1'].mean())/2,
        (df_women['fun3_s'].mean() + df_women['fun3_1'].mean())/2,
        (df_women['amb3_s'].mean() + df_women['amb3_1'].mean())/2],
      theta=categories,
      fill='toself',
      name='Women'
))
fig.update_layout(
    title={
        'text': "How men perceive themselves (red)<br> vs. <br>how women perceive themselves (blue)",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
)
fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[6.5, 8.5]
    )),
  showlegend=True
)

fig.show()


#%%
'''
What men and women say they look for
'''

categories = ['Attractivity','Sincerity','Intelligence',
           'Fun', 'Ambition']

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
     r=[ df_women['attr3_1'].mean(),
        df_women['sinc3_1'].mean(),
        df_women['intel3_1'].mean(),
        df_women['fun3_1'].mean(),
        df_women['amb3_1'].mean()],
      theta=categories,
      fill='toself',
      name='Initial'
))

fig.add_trace(go.Scatterpolar(
     r=[ df_women['attr3_s'].mean(),
        df_women['sinc3_s'].mean(),
        df_women['intel3_s'].mean(),
        df_women['fun3_s'].mean(),
        df_women['amb3_s'].mean()],
      theta=categories,
      fill='toself',
      name='Halfway through'
))


fig.update_layout(
    title={
        'text': "How women perceive themselves initially (red)<br> vs. <br>how women perceive themselves halfway through (blue)",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    paper_bgcolor = 'pink'
)
fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[6.5, 8.5]
    )),
  showlegend=True
)

fig.show()



#%%
color = ["Red", "Blue", "Orange", "Yellow", "Green", "Grey"]
sns.set_palette(color)


#%%
sns.scatterplot(x=df_individuals['intel3_1'], y=df_individuals['intel'],hue=df_individuals['gender'])

## TODO
## graphique évolution what you look for en fonction de l'age
## radar chart what you look for au début et au milieu


#%%
sns.boxplot(x=df_individuals['intel3_1'], y=df_individuals['intel'],hue=df_individuals['gender'])

#%%
sns.displot(x=df_individuals['you_call'], hue=df_individuals['gender'])
#%%
sns.displot(x=df_individuals['them_cal'], hue=df_individuals['gender'])


#%%
sns.displot(x=df_individuals['intel3_1'],hue=df_individuals['gender'])

#%%
sns.displot(x=df_individuals['intel'],hue=df_individuals['gender'])

#%%
sns.displot(x=df_individuals['race'])

#%%
sns.displot(x=df_individuals['dec_o'],hue=df_individuals['gender'])
plt.show()

#%%
sns.set(rc={"figure.figsize":(8, 8)})
sns.displot(x=df['age'], hue = df['gender'], bins=15).savefig("distagebygender.png")
#fig = agegender_plot.get_figure()
#fig.savefig("agebygender.png") 

#%%
sns.set(rc={"figure.figsize":(8, 2)})
agegender_plot = sns.boxplot(y=df['gender'], x=df['age'], hue = df['gender'])
fig = agegender_plot.get_figure()
fig.savefig("boxplotagebygender.png") 




#%%
df.groupby('gender').boxplot(column=['attr1_1'], subplots=False)
#%%
df.groupby('gender').boxplot(column=['intel1_1'], subplots=False)
#%%
df.groupby('gender').boxplot(column=['fun1_1'], subplots=False)
#%%
df.groupby('gender').boxplot(column=['amb1_1'], subplots=False)
#%%
df.groupby('gender').boxplot(column=['shar1_1'], subplots=False)


#%%
print('Individu médian recherché par chaque sexe')
print('Attractivité')
print(df.groupby('gender')['attr1_1'].median())
print('Intelligence')
print(df.groupby('gender')['intel1_1'].median())
print('Fun')
print(df.groupby('gender')['fun1_1'].median())
print('Ambition')
print(df.groupby('gender')['amb1_1'].median())
print('Même centres d''intérêt')
print(df.groupby('gender')['shar1_1'].median())


#%%
median_total_F = df[df['gender']=='F'].shar1_1.median() + df[df['gender']=='F'].attr1_1.median() + df[df['gender']=='F'].intel1_1.median() + df[df['gender']=='F'].fun1_1.median() + df[df['gender']=='F'].amb1_1.median()
print('Somme des médianes de chaque caractéristique recherchées (Femmes)')
print(median_total_F)
#%%
median_total_M = df[df['gender']=='M'].shar1_1.median() + df[df['gender']=='M'].attr1_1.median() + df[df['gender']=='M'].intel1_1.median() + df[df['gender']=='M'].fun1_1.median() + df[df['gender']=='M'].amb1_1.median()
print('Somme des médianes de chaque caractéristique recherchées (Hommes)')
print(median_total_M)

#%%

# Plot the heatmap and annotation on it
df_Corr = df.corr().abs()
sns.heatmap(df_Corr, xticklabels=df_Corr.columns, yticklabels=df_Corr.columns, cmap='PiYG')

#%%
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
df_sorted_corr = get_top_abs_correlations(df, df.shape[1])

#%%
#Scatter plot
sns.scatterplot(x="R&D Spend", y="Profit", data=df)


#%%
#PCA
pca = PCA(n_components =10)
df_PCA = pca.transform(df)

#%%
#Scatter plot with linear regression
sns.regplot(x="R&D Spend", y="Profit", data=df)

#%%
#Scatter plot with polynomial regression
sns.lmplot(x="R&D Spend", y="Profit",  order=2, data=df)



#%%

# Separate target variable Y from features X
print("Separating labels from features...")
features_list = ["R&D Spend", "Administration", "Marketing Spend", "State"]
X = df.loc[:,features_list]
y = df.loc[:,"Profit"]
print("...Done.")

# Divide dataset Train set & Test set 
## First we import train_test_split

#%%

print("Splitting dataset into train set and test set...")
## Then we use train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Allows you to stratify your sample.
# Meaning, you will have the same proportion of categories in test and train set
print("...Done.")

# Missing values
print("Imputing missing values...")
print(X_train)
print()
imputer = SimpleImputer(strategy="mean") # Instanciate class of SimpleImputer with strategy of mean
# Other strategies : median, most_frequent, constant
# Also exists : IterativeImputer (regression)
X_train = X_train.copy() # Copy dataset to avoid caveats of assign a copy of a slice of a DataFrame
# More info here https://towardsdatascience.com/explaining-the-settingwithcopywarning-in-pandas-ebc19d799d25

X_train.iloc[:,[1,2]] = imputer.fit_transform(X_train.iloc[:,[1,2]])
# Fit and transform columns where there are missing values
print("...Done!")
print(X_train) 

#%%

# Encoding categorical features and standardizing numeric features
print("Encoding categorical features and standardizing numerical features...")

numeric_features = [0, 1, 2]
numeric_transformer = StandardScaler()
# Also exists : MinMaxScaler()

categorical_features = [3]
categorical_transformer = OneHotEncoder()

featureencoder = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),    
        ('num', numeric_transformer, numeric_features)
        ]
    )
X_train = featureencoder.fit_transform(X_train)
print("...Done.")
print(X_train[:5])
# Print first 5 rows (not using iloc since now X_train became a numpy array)

#%%

# Encoding labels
print("Encoding labels...")
print(y_train)

labelencoder = LabelEncoder()
Y_train = labelencoder.fit_transform(y_train)
print("...Done.")
print(Y_train[:5])
# Print first 5 rows (not using iloc since now y_train became a numpy array)


#%%

# Train model
print("Train model...")
regressor = LinearRegression()
regressor.fit(X_train, y_train) # This steps is the actual training !
print("...Done.")

#%%

# Predictions on training set
print("Predictions on training set...")
y_train_pred = regressor.predict(X_train)
print("...Done.")
print(y_train_pred[:5]) # print first 5 rows (not using iloc since now y_train became a numpy array)
print()

### Testing pipeline ###
print("--- Testing pipeline ---")

# Standardizing numeric features
print("Standardizing numerical features...")
print(X_test)
print()

X_test = featureencoder.transform(X_test)

print("...Done.")
print(X_test[:5]) # print first 5 rows (not using iloc since now X_test became a numpy array)
print()

# Predictions on test set
print("Predictions on test set...")
y_test_pred = regressor.predict(X_test)
print("...Done.")
print(y_test_pred[:5])
print()

#%%

# Performance assessment
print("--- Assessing the performances of the model ---")

# Print R^2 scores
print("R2 score on training set : ", regressor.score(X_train, y_train))
print("R2 score on test set : ", regressor.score(X_test, y_test))

# Print RMSE score
rms_score = sqrt(mean_squared_error(y_test, y_test_pred))
print("RMSE:", rms_score)

# Print MAE score
mae_score = mean_absolute_error(y_test, y_test_pred)
print(mae_score)

print("coefficients are: ", regressor.coef_) 
print("Constant is: ", regressor.intercept_)

# Access transformers from feature_encoder
print("All transformers are: ", featureencoder.transformers_)

# Access one specific transformer
print("One Hot Encoder transformer is: ", featureencoder.transformers_[0][1])

# Print categories
categorical_column_names = featureencoder.transformers_[0][1].categories_
print("Categorical columns are: ", categorical_column_names)

numerical_column_names = X.iloc[:, numeric_features].columns # using the .columns attribute gives us the name of the column 
print("Numerical columns are: ", numerical_column_names)

# Append all columns 
all_column_names = np.append(categorical_column_names, numerical_column_names)

# Feature importance 
feature_importance = pd.DataFrame({
    "feature_names": all_column_names,
    "coefficients":regressor.coef_
})

# Set coefficient to absolute values to rank features
feature_importance["coefficients"] = feature_importance["coefficients"].abs()

# Visualize ranked features using seaborn
sns.set_style("darkgrid")
sns.color_palette("Set2")
sns.catplot(x="feature_names", 
            y="coefficients", 
            data=feature_importance.sort_values(by="coefficients", ascending=False), 
            kind="bar",
            aspect=16/9) # Resize graph
# %%
