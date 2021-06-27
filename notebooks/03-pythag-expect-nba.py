#!/usr/bin/env python
# coding: utf-8

# # Week 1 - Pythagorean Expectation on NBA data (2018 season)

# In[27]:


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

# Download data from Kaggle
# 
# https://www.kaggle.com/nathanlauga/nba-games/activity

# In[5]:


data_dir = Path('../data/raw/wk1-nba/nba-kaggle')
[f for f in data_dir.iterdir()]


# In[28]:


game_details = pd.read_csv(data_dir / 'games_details.csv').pipe(lower_case_col_names)
games = pd.read_csv(data_dir / 'games.csv').pipe(lower_case_col_names)
teams = pd.read_csv(data_dir / 'teams.csv').pipe(lower_case_col_names)
players = pd.read_csv(data_dir / 'players.csv').pipe(lower_case_col_names)
rankings = pd.read_csv(data_dir / 'ranking.csv').pipe(lower_case_col_names)

rankings.head()players.head()game_details.head()
# In[85]:


teams.head()


# In[29]:


print(games.shape)
games.head()


# In[31]:


games_18 = games.query("season == 2018")

print(games_18.shape)
games_18.head()


# In[55]:


tmp = (
    games_18[
        [
            "game_date_est",
            "game_id",
            "home_team_id",
            "visitor_team_id",
            "pts_home",
            "pts_away",
            "home_team_wins",
        ]
    ]
    .rename(columns={"game_date_est": "date",})
    .assign(away_team_wins=lambda x: 1 - x['home_team_wins'])
    #  .merge(teams[['team_id', 'abbreviation', 'nickname', 'city']], left_on='home_team_id', right_on='team_id')
)


# Check for missing rows

# In[51]:


print(tmp.shape, f"missing data? {tmp.shape != tmp.dropna().shape}")


# In[56]:


tmp


# In[73]:


home_cols = ['date', 'game_id', 'home_team_id', 'pts_home', 'pts_away', 'home_team_wins', 'away_team_wins',]
away_cols = ['date', 'game_id', 'visitor_team_id', 'pts_away', 'pts_home', 'away_team_wins', 'home_team_wins', ]
home_tall = (tmp[home_cols]
  .rename(columns={'home_team_id': 'team_id', 
                  'home_team_wins': 'W',
                   'away_team_wins': 'L',
                  'pts_home': 'pts_scored',
                  'pts_away': 'pts_allowed'}))
away_tall = (tmp[away_cols]
  .rename(columns={'visitor_team_id': 'team_id', 
                  'away_team_wins': 'W',
                   'home_team_wins': 'L',
                  'pts_away': 'pts_scored',
                  'pts_home': 'pts_allowed'}))


# In[74]:


home_tall.head()


# In[75]:


away_tall.head()


# In[93]:


team_id_names = teams.assign(name=lambda x: x["city"] + " " + x["nickname"])[
    ["team_id", "name"]
]
team_id_names.head()


# Note: I didn't pull out playoff games. Normally, each team plays 82 games

# In[95]:


pyth_ex_nba = (
    pd.concat([home_tall, away_tall])
    .merge(team_id_names, on='team_id')
    .groupby("name")["W", "L", "pts_scored", "pts_allowed"]
    .sum()
    .assign(
        wpc=lambda x: x["W"] / (x["W"] + x["L"]),
        pyth=lambda x: x["pts_scored"] ** 2
        / (x["pts_scored"] ** 2 + x["pts_allowed"] ** 2),
    )
)

pyth_ex_nba


# In[96]:


sns.relplot(x='pyth', y='wpc', data=pyth_ex_nba)


# # Regression

# In[97]:


pyth_lm = smf.ols(formula = 'wpc ~ pyth', data=pyth_ex_nba).fit()
pyth_lm.summary()


# Interpretation
# 
# * std err: 0.259 (low)
# * t-score: 24.5 (high)
# * P-score: 0 << 0.05, so VERY confident this coef is statistically significantly different from 0
#     

# In[ ]:




