#!/usr/bin/env python
# coding: utf-8

# # Week 1

# # Imports

# In[7]:


# %load /Users/bartev/dev/github-bv/sporty/notebooks/imports.py
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


# In[8]:


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


# # Read data

# In[9]:


pwd


# In[15]:


with open('../data/raw/wk1-baseball/retrosheet-gamelog-header.txt', 'r') as f:
    retro_cols = f.read().split(',')


# In[17]:


len(retro_cols)


# In[21]:


MLB = pd.read_csv('../data/raw/wk1-baseball/GL2018.csv', names=retro_cols)
MLB


# In[23]:


MLB18 = (MLB
        [['VisitingTeam', 'HomeTeam', 'VisitorRunsScored','HomeRunsScore', 'Date']]
        .rename(columns={'VisitorRunsScored': 'VisR', 'HomeRunsScore': 'HomeR'}))
MLB18


# In[27]:


# Assign home/away win

MLB18 = MLB18.assign(
    hwin=lambda x: np.where(x["HomeR"] > x["VisR"], 1, 0),
    awin=lambda x: np.where(x["HomeR"] < x["VisR"], 1, 0),
    count=1,
)


# In[28]:


MLB18


# In[33]:


MLBhome = (
    MLB18.groupby("HomeTeam")["hwin", "HomeR", "VisR", "count"]
    .sum()
    .reset_index()
    .rename(
        columns={"HomeTeam": "team", "VisR": "VisRh", "HomeR": "HomeRh", "count": "Gh"}
    )
)
MLBhome


# In[34]:


MLBaway = (
    MLB18.groupby("VisitingTeam")["awin", "HomeR", "VisR", "count"]
    .sum()
    .reset_index()
    .rename(
        columns={"VisitingTeam": "team", "VisR": "VisRa", "HomeR": "HomeRa", "count": "Ga"}
    )
)
MLBaway


# In[37]:


MLB18 = pd.merge(MLBhome, MLBaway, on='team')
MLB18


# In[ ]:




