#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

from src.utils import lower_case_col_names, drop_suffix
import src.data.load_data as ld
from src.data.load_data import get_nba_game_team_points, load_nba, load_nba_games_dataset


# # Summary Statistics

# Central tendency
# * mean
# * median
# * mode
# 
# Variation
# * variance
# * standard deviation
# * coefficient of variation (std dev / mean. between 0 and 1)
# 

# In[12]:


nba_games = load_nba_games_dataset()


# In[7]:


nba_games.head()


# In[13]:


nba_games.dtypes


# In[14]:


nba_games.describe(include='all')


# In[15]:


nba_games['pts'].describe()


# In[16]:


nba_games['pts'].mean()


# In[17]:


nba_games['pts'].median()


# In[18]:


nba_games['pts'].std()


# In[20]:


nba_games.groupby(['wl']).mean()


# In[22]:


nba_games['game_date_est'].describe()


# # Visualizing data

# ## Histogram

# In[48]:


nba_games.hist(
    column="pts",
#     by="wl",   # groupby
    grid=True,
    sharex=True,
#     layout=(2, 1),  # arrangement of axes
#     figsize=(5, 10),
    bins=20,
    rwidth=0.9,  # leave space between bins
)


# In[ ]:




