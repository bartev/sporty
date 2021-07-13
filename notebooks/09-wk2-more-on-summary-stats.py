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


# # Import updated NBA Game data

# In[41]:


nba_games = load_nba_games_dataset()
nba_games.head()


# # Central tendency vs variation
# 
# Compare the success rates of 2 point FG and 3 pt FG to demonstrate the difference between central tendency and variaion

# ## Calculate the summary statistics for the percentages of 2 pt FG and 3 pt FG

# In[4]:


nba_games['fg_pct'].describe()


# In[5]:


nba_games['fg3_pct'].describe()


# ## Compare the distribution of 2pt and 3pt FG percentage using a histogram

# ### Side by side

# In[12]:


nba_games.hist(column=['fg_pct', 'fg3_pct'], bins=20, sharex=True, sharey=True, grid=True)


# ### Plot 2 histograms in the same graph in different colors

# * Use `plot.hist` instead of `hist`
# * add title and axis labels using `plt.title` and `plt.xlabel`, `plt.ylabel`

# In[10]:


nba_games[['fg_pct', 'fg3_pct']].plot.hist(alpha=0.3, bins=20)
plt.xlabel('Field Goal Percentage')
plt.ylabel('Frequency')
plt.title('Distribution of Field Goal Percentages', fontsize=15)
plt.savefig('fg_pct_distrib.png')


# ## Make graphs grouped by criteria

# In[22]:


nba_games.hist(by='wl', column='fg_pct', color='red', bins=15, sharex=True, sharey=True, alpha=0.6, grid=True)


# # Create time series graph
# 
# First, change the data type of 'game_date' to `datetime`

# In[25]:


nba_games.dtypes


# In[43]:


nba_games_d = nba_games.assign(game_date = lambda x: pd.to_datetime(x['game_date']))
nba_games_d.head()


# In[44]:


nba_games_d['season'].value_counts()


# In[45]:


pistons_games = nba_games_d[(nba_games_d['nickname'] == 'Pistons')]
pistons_games


# In[47]:


pistons_games.plot(x='game_date', y='pts')


# In[ ]:




