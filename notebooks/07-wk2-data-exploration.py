#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[29]:


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


# In[100]:


nba_games = load_nba_games_dataset()
nba_games.head()


# # Explore the dataset

# ## Qualitative vs Quantitative Data

# * object: qualitative variable
# * int64: quantitative and discrete (integer) (-2^63) - (2^63 - 1)
# * float64: quantitative and continuous - real numbers (64 bit)

# In[81]:


nba_games.dtypes


# ## Convert a categorical variable to a dummy variable

# `pd.get_dummies` creates new columns, and drops the original columns.

# In[84]:


nba_games.head()


# In[88]:


dummy = pd.get_dummies(nba_games, columns=['wl'])
dummy.columns


# ### 3 ways to merge `wl_W`

# In[91]:


pd.concat([nba_games, dummy['wl_W']], axis=1).rename(columns={'wl_W': 'win'}).head()


# In[93]:


nba_games.pipe(lambda x: pd.concat([x, dummy['wl_W']], axis=1))


# In[101]:


nba_games_w = nba_games.merge(dummy).drop(columns='wl_L').rename(columns={'wl_W':'win'})
nba_games_w


# In[102]:


nba_games['game_date_est'].dtype


# Currently, the dates are stored as objects (so treated equally(???) w/o any ordering)

# Use `pd.to_datetime()` to convert to a date variable

# In[104]:


import datetime
nba_games_d = (nba_games_w
               .assign(game_date=lambda x: pd.to_datetime(x['game_date_est']))
               .drop(columns='game_date_est'))
nba_games_d.dtypes


# In[105]:


nba_games_d.head()


# In[ ]:




