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


# # Correlation Analyses
# 
# * Any relationship between 2 variables?
# 
# * Start with a scatterplot

# In[3]:


nba_games = load_nba_games_dataset()


# In[4]:


nba_games.plot.scatter(x='ast', y='fgm')


# Add a regression line to better visualize the relationship
# 
# Use Seaborn

# In[6]:


sns.regplot(x='ast', y='fgm', data=nba_games, marker='.')
plt.xlabel('Assists')
plt.ylabel('Field Goals Made')
plt.title('Relationship between the nmber of assists and FG made', fontsize=15)


# Quantify the relation ship using correlation analysis.
# 
# Correlation coefficient - single number to summarize the relationship

# # Covariance

# A measure of the joint variability of 2 random variables.
# 
# The sign of the covariance shows the tendency in the linear relationship between the variables.
# 
# $$
# \sigma_{xy} = cov(x, y) = E[(x - E(x))(y - E(y))]
# $$

# # Interpreting covariance
# 
# * Covariance depends on the unit of measurement.
# * Typically, we only look at the *sign* of the covariance.

# # Correlation Coefficient
# 
# Covariance divided by standard deviations.
# 
# * Correlation coefficient measures the lineaer correlation between 2 variables.
# * does not depend on units
# * values are between -1 and 1
# 
# $$
# r = \frac{cov(x, y)}{\sigma_x \sigma_y}
# $$

# Correlation between assists and field goals made

# In[7]:


nba_games['ast'].corr(nba_games['fgm'])


# In[9]:


sns.regplot(x='ast', y='fga', data=nba_games, marker='.')
plt.xlabel('Assists')
plt.ylabel('Field Goals Attempted')
plt.title('Relationship between the nmber of assists and FG attempted', fontsize=15)


# In[10]:


nba_games['ast'].corr(nba_games['fga'])


# # Plot scatter plot by group using `hue` option

# In[ ]:


* `lmplot` combines `regplot` and `FacetGrid`
* `FacetGrid` produces a multi-plot grid for plotting conditional relationships.


# In[12]:


sns.lmplot(x='ast', y='fga', hue='wl', data=nba_games)
plt.xlabel('Assists')
plt.ylabel('Field Goals Made')
plt.title('Relationship between the number of assists and FG made', fontsize=15)


# # Construct a correlation table for all numeric variables

# In[14]:


nba_games.corr(method='pearson')


# In[15]:


nba_games.dtypes


# In[ ]:




