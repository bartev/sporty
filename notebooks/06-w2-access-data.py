#!/usr/bin/env python
# coding: utf-8

# # Accessing Data in Python (part 2)

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

from src.utils import lower_case_col_names
import src.data.load_data as ld
from src.data.load_data import get_nba_game_team_points


# In[3]:


nba_games = ld.load_nba('games')
nba_games.head()


# In[4]:


games_18 = nba_games.query("season == 2018")


# In[5]:


games_18.info()


# ## Use `info()` to check for missing variables

# In[6]:


nba_games.info()


# Note: some variables have missing data

# ## Use `isnull()`, `notnull()` to look for missing values

# this doesn't work - why?

# In[7]:


nba_games[nba_games.isnull()]


# This doesn't seem to do anything either

# In[8]:


nba_games[nba_games.notnull()]


# # Handling Missing Values

# ## Drop observations with missing values in the variable `fg_pct_home`

# Using `notnull` on a single column works, though.

# In[29]:


nba_games[pd.notnull(nba_games['fg_pct_home'])]


# Call `notnull` either as a method on the column, or as a function from pandas

# In[31]:


nba_games[nba_games['fg_pct_home'].notnull()]


# In[28]:


nba_games[pd.isnull(nba_games['fg_pct_home'])].shape


# ## Data imputation

# ### `fillna`

# In[32]:


nba_games.mean()


# In[35]:


mean_filled = nba_games.fillna(nba_games.mean())
print(f'shape: {mean_filled.shape}')
mean_filled.head()


# ## Create variables

# In[38]:


nba_games[['pts_home', 'pts_away']].head()


# In[40]:


(nba_games['pts_home'] + nba_games['pts_away']).head()


# ### Based on a condition

# Could use `home_team_wins`, but I'm doing it this way to make sure it's all consistent.m

# In[44]:


(nba_games
 .head()
 .fillna(lambda x: x.mean())
 [['pts_home', 'pts_away', 'home_team_wins']]
 .assign(result=lambda x: np.where(x['pts_home'] > x['pts_away'], 'W', 'L'))
)


# Drop the newly created variable

# In[46]:


(nba_games
 .head()
 .fillna(lambda x: x.mean())
 [['pts_home', 'pts_away', 'home_team_wins']]
 .assign(result=lambda x: np.where(x['pts_home'] > x['pts_away'], 'W', 'L'))
 .drop('result', axis=1) 
)


# In[47]:


nba_games.head()


# ### Create a variable based on a group

# Download 

# In[20]:


nba_gtp = get_nba_game_team_points()
nba_gtp


# * 2 observations: 1 each for home and away teams
# * Create a variable `point_diff`.
# 
# 1. sort by `game_id` and `wl`. (puts the same game rows together, with winning team first)
# 2. the `groupby` and `diff` will give the diff between the rows in the same match

# In[30]:


(nba_gtp.sort_values(["game_id", "wl"])[["game_id", "hv", "points"]]
    .assign(point_diff=lambda x: x.groupby(["game_id"])["points"].diff()))


# `point_diff` will only have the point difference for the winning team (not the losing team)
# 
# Fill in the missing values for the losing team with the mean of the observation.
# 
# Use the `transform` function to map the values to the same index of the original data frame.

# In[29]:


tmp = (
    nba_gtp.sort_values(["game_id", "wl"])[["game_id", "hv", "points"]]
    .assign(point_diff=lambda x: x.groupby(["game_id"])["points"].diff())
    .assign(
        point_diff=lambda x: x["point_diff"].fillna(
            x.groupby("game_id")["point_diff"].transform("mean")
        )
    )
    .dropna()
)
print(f"tmp.shape: {tmp.shape}")
tmp


# In[117]:


# `transform` works on a grouped 
tmp.groupby('game_id')['point_diff'].transform('mean')


# In[119]:


display(tmp)


# # Create a new dataframe

# ## Number of games per season by team

# Create a variable that equals the total number of observations in a group using `size`

# In[129]:


games_dataset = (
    get_nba_game_team_points()
    .groupby(["team_id", "season"])
    .size()
    .reset_index(name="game_count")
)
games_dataset


# In[ ]:




