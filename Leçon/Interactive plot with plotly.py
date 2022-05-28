# Interactive plot with plotly
# https://app.jedha.co/course/interactive-graphs-pt/interactive-plot-with-plotly-pt


#%%
import plotly.io as pio

pio.renderers.default = "browser"  # If you are on the workspaces: pio.renderers.default = "iframe_connected"

# Create dictionnary
fig = {
    'data': [{
        "type": "scatter",
        "x": [1, 2, 3, 4],
        "y": [2, 3, 2, 4]
    }],

    "layout": {
        "title": {
            "text": "This figure has been made by a python dictionnary"
        }
    }
}

# Show figure 
pio.show(fig)

#%%
import plotly.graph_objects as go



fig = go.Figure(
    data=[go.Scatter(x=[1, 2, 3, 4], y=[2, 3, 2, 4])],
    layout=go.Layout(
        title=go.layout.Title(text="This figure has been made by a plotly graph object")
    )
)

fig.show(renderer='vscode')
# %%

import plotly.express as px
df = px.data.iris()
df.head()
fig = px.treemap(df, 
                 path=['sepal_length','sepal_width','petal_length','petal_width'], 
                 values='petal_width',
                 color='species'
                )
fig.show()

# %%
import plotly.express as px
df = px.data.iris()
df.head()
fig = px.parallel_categories(df, color="species_id", 
                             color_continuous_scale=px.colors.sequential.Inferno)
fig.show()


# %%

fig = px.sunburst(df, path=['sepal_length','sepal_width','petal_length','petal_width'], values='species_id',color='species_id')
fig.show()


#%%
# Create categories 
df = px.data.tips()
fig = px.scatter(df, x="total_bill", y="tip", facet_row="time", facet_col="day", color="smoker", trendline="ols",
          category_orders={"day": ["Thur", "Fri", "Sat", "Sun"], "time": ["Lunch", "Dinner"]})
fig.show()
# %%
df = px.data.election()
fig = px.scatter_3d(df, x="Joly", y="Coderre", z="Bergeron", color="winner", size="total", hover_name="district",
                  symbol="result", color_discrete_map = {"Joly": "blue", "Bergeron": "green", "Coderre":"red"})
fig.show()


# %%
df = px.data.gapminder()
fig = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
                    size="pop", color="continent", hover_name="country",
                    log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])

fig.show()
# %%
from plotly.offline import plot

df = px.data.gapminder()
fig=px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
           size="pop", color="continent", hover_name="country",
           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])
plot(fig)

# %%
