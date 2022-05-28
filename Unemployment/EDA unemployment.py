#%%
import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib as plt


#%%
df_unemployment = pd.read_csv('fdeec17.csv')
df_unemployment.head()



#%%
df_unemployment['ACTEU_type'] = df_unemployment['ACTEU'].apply(lambda x :   'Actif' if x == 1 else
                                                                        'Chômeur' if x == 2 else
                                                                        'Inactif' if x== 3 else
                                                                        'Pas de données')

# %%

labels = ['Actif', 'Chômeur', 'Inactif']
sizes = [   df_unemployment.loc[df_unemployment['ACTEU'] == 1, 'ACTEU'].count(),
            df_unemployment.loc[df_unemployment['ACTEU'] == 2, 'ACTEU'].count(),
            df_unemployment.loc[df_unemployment['ACTEU'] == 3, 'ACTEU'].count()]
colors = ['yellowgreen', 'gold', 'lightskyblue']
plt.pyplot.pie(sizes, colors=colors, shadow=True, startangle=90, autopct='%1.1f%%')
plt.legend(labels, loc="right")
plt.axis('equal')
plt.tight_layout()
plt.show()


# %%
acteu_list = df_unemployment['ACTEU_type'].value_counts()
labels_acteu = acteu_list.index
plt.pyplot.pie(acteu_list, labels=labels_acteu, autopct='%1.1f%%')

# %%
