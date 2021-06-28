#!/usr/bin/env python
# coding: utf-8

# # Description

# Can we forecast using the Pythagorean Expectation?
# 
# 1. Divide a season into 2 halves
# 2. Take Pyth-Expec from 1st half, and see how well it fits with the win percentage in the 2nd half.
# 3. Control: how well does the wpc for the 1st half fit for the 2nd half?

# In[3]:


# %load ./imports.py
# %load /Users/bartev/dev/github-bv/sporty/notebooks/imports.py

## Where am I
get_ipython().system('echo $VIRTUAL_ENV')

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))

# magics
get_ipython().run_line_magic('load_ext', 'blackcellmagic')
# start cell with `%%black` to format using `black`

get_ipython().run_line_magic('load_ext', 'autoreload')
# start cell with `%autoreload` to reload module
# https://ipython.org/ipython-doc/stable/config/extensions/autoreload.html

# reload all modules when running
get_ipython().run_line_magic('autoreload', '2')

# imports

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns

from importlib import reload
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# https://plotnine.readthedocs.io/en/stable/

import plotnine as p9
from plotnine import ggplot, aes, facet_wrap

from src.utils import lower_case_col_names


# # Read data

# In[4]:


data_dir = Path('../data/raw/wk1-baseball/')
with open(data_dir / 'retrosheet-gamelog-header.txt', 'r') as f:
    retro_cols = f.read().split(',')
MLB = pd.read_csv(data_dir / 'GL2018.csv', names=retro_cols)


# In[6]:


print(MLB.shape)
MLB.head()


# In[10]:


mlb18 = (
    MLB[["VisitingTeam", "HomeTeam", "VisitorRunsScored", "HomeRunsScore", "Date"]]
    .rename(columns={"VisitorRunsScored": "VisR", "HomeRunsScore": "HomR"})
    .assign(count=1)
)
mlb18

# Performance when home team
mlb18.melt(value_vars=['VisitingTeam', 'HomeTeam'], id_vars=['VisR', 'HomR', 'Date', 'count'])
# In[18]:


# Performance when home team
mlb_home = (
    mlb18[["HomeTeam", "HomR", "VisR", "count", "Date"]]
    .assign(home=1)
    .rename(columns={"HomeTeam": "team", "VisR": "RA", "HomR": "R"})
)
mlb_home


# In[19]:


# Performance when away team
mlb_away = (
    mlb18[["VisitingTeam", "VisR", "HomR", "count", "Date"]]
    .assign(home=0)
    .rename(columns={"VisitingTeam": "team", "VisR": "R", "HomR": "RA"})
)
mlb_away


# In[26]:


mlb = (pd.concat([mlb_home, mlb_away])
      .assign(win=lambda x: np.where(x['R'] > x['RA'], 1, 0)))
print(mlb.shape)
mlb.head()


# ## Split season midway (at the date of the All Star game)
# 
# 2018-07-17

# In[27]:


half1 = mlb.query("Date < 20180717")
half2 = mlb.query("Date >= 20180717")


# In[28]:


half1.describe()


# In[29]:


half2.describe()


# ## Performance variables

# In[32]:


half1_perf = (half1.groupby('team')['count', 'win', 'R', 'RA'].sum().reset_index()
             .rename(columns={'count':'count1', 'win':'win1', 'R':'R1', 'RA':'RA1'})
             .assign(wpc1=lambda x: x['win1'] / x['count1'],
                    pyth1=lambda x: x['R1']**2 / (x['R1']**2 + x['RA1']**2)))
half1_perf


# In[35]:


half2_perf = (half2.groupby('team')['count', 'win', 'R', 'RA'].sum().reset_index()
             .rename(columns={'count':'count2', 'win':'win2', 'R':'R2', 'RA':'RA2'})
             .assign(wpc2=lambda x: x['win2'] / x['count2'],
                    pyth2=lambda x: x['R2']**2 / (x['R2']**2 + x['RA2']**2)))
half2_perf


# In[36]:


half2_predictor = pd.merge(half1_perf, half2_perf, on='team')
half2_predictor.head()


# # Plot performance

# In[37]:


sns.relplot(x='pyth1', y='wpc1', data=half2_predictor)


# In[38]:


sns.relplot(x='wpc1', y='wpc2', data=half2_predictor)


# # Correlation between key variables

# In[39]:


keyvars = half2_predictor[['team', 'wpc2', 'wpc1', 'pyth1', 'pyth2']]


# In[40]:


keyvars.corr()


# ## Interpretation
# 
# * variable of interest on row index
# * what's the correlation between the wpc2 and the wpc1, and between wpc2 and pyth1?
# * correlation between wpc2 and pyth1 is HIGHER than that between wpc2 and wpc1.
# 
# 
# This is why Bill James proposed pythagorean expectation is a better predictor than win percentage for future performance

# In[42]:


keyvars.sort_values(by=['wpc2'], ascending=False)


# In[44]:


sns.relplot(y='wpc2', x='pyth1', data=keyvars)


# In[ ]:




