# https://app.jedha.co/course/interactive-graphs-pt/earthquakes-pt
# Interactive Graphs


#%%
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pandas as pd 

#%%
df_earthquakes = pd.read_csv('earthquakes.csv')
df_earthquakes.head()
# %%
df_earthquakes["datetime"]  = df_earthquakes["Date"] + " " + df_earthquakes["Time"]

#%%
df_earthquakes["datetime"] = pd.to_datetime(df_earthquakes["datetime"])
# %%
df_earthquakes.to_csv('earthquakes_datetimes.csv')

#%%
df_earthquakes.describe(include='all')

#%%
df_earthquakes = df_earthquakes.sort_values(by='datetime')

# %%
pio.renderers.default = 'notebook_connected'
px.histogram(df_earthquakes,'datetime')

# %%
fig = go.Figure(
    data = go.Histogram(
        x = df_earthquakes['Date'], nbinsx = df_earthquakes['Date'].nunique()),
    layout = go.Layout(
        title = go.layout.Title(text = "Number of observations per day", x = 0.5),
        xaxis = go.layout.XAxis(title = 'X', rangeslider = go.layout.xaxis.Rangeslider(visible = True))
    )
)

fig.show()

# %%
