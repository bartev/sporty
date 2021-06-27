#!/usr/bin/env python
# coding: utf-8

# # Week 1
# 
# IPL (Indian Premier League) (cricket)

# # Imports

# In[1]:


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


# In[2]:


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


# In[30]:


get_ipython().system('pip install xlrd')


# In[32]:


get_ipython().system('pip install openpyxl')


# # Read data

# Need `IPL2018teams` data
# 
# * https://www.kaggle.com/manasgarg/ipl

# ## archive
# 
# 2008-2017
data_dir = Path('/Users/bartev/dev/github-bv/sporty/data/raw/wk1-IPL/archive')
matches = pd.read_csv(data_dir / 'matches.csv')

deliveries = pd.read_csv(data_dir / 'deliveries.csv')deliveries.head()matches.groupby('season').size()
# ## archive 2
# 2008 - 2019
data_dir2 = Path('/Users/bartev/dev/github-bv/sporty/data/raw/wk1-IPL/archive 2')
matches2 = pd.read_csv(data_dir2 / 'matches.csv')
deliveries2 = pd.read_csv(data_dir2 / 'deliveries.csv')matches2.groupby('season').size()deliveries2
# ## archive-3
# 
# I found this data somewhere on Kaggle (didn't keep the link
# 
# More data than `archive 2`?

# In[157]:


data_dir = Path('/Users/bartev/dev/github-bv/sporty/data/raw/wk1-IPL/archive-3')
matches = pd.read_csv(data_dir3 / 'matches.csv')
deliveries = pd.read_csv(data_dir3/ 'deliveries.csv')
teams = pd.read_csv(data_dir3 / 'teams.csv')
teamwise = pd.read_csv(data_dir3 / 'teamwise_home_and_away.csv')
players = pd.read_excel(data_dir3 / 'Players.xlsx')


# In[158]:


matches.groupby('Season').size()


# In[63]:


deliveries.head()


# In[64]:


matches.head(2)


# In[159]:


teamwise


# In[ ]:


ipl18 = matches.query("Season == 'IPL-2018'").assign(
    hwin=lambda x: np.where(x["team1"] == x["winner"], 1, 0),
    awin=lambda x: np.where(x["team2"] == x["winner"], 1, 0),
    #                 htruns=lambda x: np.where(x['team1'] == x[''])
)
ipl18


# In[101]:


# get the match ids for season 2018
match_ids_2018 = ipl18["id"]

# Count runs scored by each team for season 2018
match_team_runs = (
    deliveries[["match_id", "inning", "batting_team", "bowling_team", "total_runs"]]
    .merge(match_ids_2018, left_on="match_id", right_on="id")
    .groupby(["match_id", "batting_team"])
    .agg({"total_runs": sum})
    .reset_index()
    .rename(columns={"match_id": "id"})
)
match_team_runs.head(5)


# In[133]:


ipl_cols = [
    "id",
    "Season",
    "date",
    "team1",
    "team2",
    "winner",
    "win_by_runs",
    "win_by_wickets",
    "hwin",
    "awin",
]

ipl18_exp = (
    ipl18[ipl_cols]
    .rename(columns={"team1": "home_team", "team2": "away_team"})
    .merge(
        match_team_runs, left_on=["id", "home_team"], right_on=["id", "batting_team"]
    )
    .drop(columns="batting_team")
    .rename(columns={"total_runs": "htruns"})
    .merge(
        match_team_runs, left_on=["id", "away_team"], right_on=["id", "batting_team"]
    )
    .drop(columns="batting_team")
    .rename(columns={"total_runs": "atruns"})
    .assign(
        my_hwin=lambda x: np.where(x["home_team"] == x["winner"], 1, 0),
        my_awin=lambda x: np.where(x["away_team"] == x["winner"], 1, 0),
    )
    #     .assign(match=lambda x: x['hwin'] == x['my_hwin'])
    .assign(count=1)
)


# In[134]:


ipl18_exp


# * Ph/Pa : number of games played at home/away

# In[143]:


iplhome = (
    ipl18_exp.groupby("home_team")["count", "hwin", "htruns", "atruns"]
    .sum()
    .reset_index()
    .rename(
        columns={
            "home_team": "team",
            "count": "Ph",
            "htruns": "htrunsh",
            "atruns": "atrunsh",
        }
    )
)
iplhome


# In[144]:


iplaway = (
    ipl18_exp.groupby("away_team")["count", "awin", "htruns", "atruns"]
    .sum()
    .reset_index()
    .rename(
        columns={
            "away_team": "team",
            "count": "Pa",
            "htruns": "htrunsa",
            "atruns": "atrunsa",
        }
    )
)
iplaway


# In[151]:


ipl18_combo = (
    iplhome.merge(iplaway, on="team")
    # aggregate home/away data for wins, games played, and runs
    .assign(
        W=lambda x: x["hwin"] + x["awin"],
        G=lambda x: x["Ph"] + x["Pa"],
        R=lambda x: x["htrunsh"] + x["atrunsa"],
        RA=lambda x: x["atrunsh"] + x["htrunsa"],
    )
    # get win percentage and pythagorean expectation
    .assign(wpc=lambda x: x['W'] / x['G'],
           pyth=lambda x: x['R']**2 / (x['R']**2 + x['RA']**2))
)

ipl18_combo


# In[152]:


sns.relplot(x='pyth', y='wpc', data=ipl18_combo)


# # Run a regression

# In[154]:


pyth_lm = smf.ols(formula = 'wpc ~ pyth', data=ipl18_combo).fit()
pyth_lm.summary()


# Interpretation
# 
# * pyth 
#     * coef = 2.5
#     * std err = 2.077
#     * t value = 1.235
#     * P-value = 0.263 >> 0.05
#     * P-value = probability that we'd observe the value 2.5 (for the coef) by chance if the true value were 0
#     * p-values > 0.05, so values are not considered statistically significant.
#         We have no confidence in this relationship.
#     * R^2 is low (20%)

# In[ ]:




