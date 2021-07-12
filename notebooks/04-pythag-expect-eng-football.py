#!/usr/bin/env python
# coding: utf-8

# # Week 1 - Pythangorean Expectation & English Football

# In[1]:


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
# 
# Download from https://fbref.com/en/comps/9/1631/2017-2018-Premier-League-Stats

# In[2]:


data_dir = Path('../data/raw/wk1-english-football/')

data = pd.read_csv(data_dir / 'season-1718.csv').pipe(lower_case_col_names)
data = pd.read_csv(data_dir / 'england.csv')


# In[3]:


data


# In[20]:


data.query("Season == 2017").groupby(['tier', 'division']).size()


# In[21]:


data.query("Season == 2017").head()


# there are 4 divisions in English Football
# 
# 1: English Premier League

# In[4]:


cols = ["Date", "Season", "home", "visitor", "hgoal", "vgoal", "division", "result"]
engl_17 = (
    data[cols]
    .query("Season == 2017")
    .assign(
        hwin=lambda x: np.where(
            x["result"] == "H", 1, np.where(x["result"] == "A", 0, 0.5)
        ),
        awin=lambda x: np.where(
            x["result"] == "A", 1, np.where(x["result"] == "H", 0, 0.5)
        ),
        count=1,
    )
)
engl_17


# ## If I want this in log format... 
# 
# (don't go down this road)

# In[45]:


engl_17_long = engl_17.melt(
    id_vars=['Date', 'Season', 'hgoal', 'vgoal', 'division', 'result', 'hwin', 'awin', 'count'], 
#     value_vars=['home', 'visitor'], 
    value_name='team',
    var_name='home_vis')
engl_17_long


# In[47]:


(engl_17_long.groupby(['team','home_vis'])['hgoal', 'vgoal', 'hwin', 'awin', 'count']
 .sum()
 .reset_index()
#  .assign(wpc=lambda x: x['hwin'])
)


# ## Continue with wide data

# In[36]:


engl_home = (
    engl_17.groupby(["home", "division"])["count", "hwin", "hgoal", "vgoal"]
    .sum()
    .reset_index()
    .rename(
        columns={
            "home": "team",
            "count": "Ph",
            "hgoal": "FTHGh",
            "vgoal": "FTAGh",
#             'hwin': 'hwinvalue'
        }
    )
)
engl_home


# In[37]:


engl_away = (
    engl_17.groupby(["visitor"])["count", "awin", "hgoal", "vgoal"]
    .sum()
    .reset_index()
    .rename(
        columns={
            "visitor": "team",
            "count": "Pa",
            "hgoal": "FTHGa",
            "vgoal": "FTAGa",
        }
    )
)
engl_away


# In[38]:


# GF - goals scores
# GA - goals conceded

engl_summary = (
    engl_home.merge(engl_away, on="team")
    .assign(
        W=lambda x: x["hwin"] + x["awin"],
        G=lambda x: x["Ph"] + x["Pa"],
        GF=lambda x: x["FTHGh"] + x["FTAGa"],
        GA=lambda x: x["FTAGh"] + x["FTHGa"],
    )
    .assign(
        wpc=lambda x: x["W"] / x["G"],
        pyth=lambda x: x["GF"] ** 2 / (x["GF"] ** 2 + x["GA"] ** 2),
    )
)
engl_summary


# # Plot

# In[82]:


sns.relplot(x='pyth', y='wpc', data=engl_summary, hue='division')


# # Regression

# In[84]:


pyth_lm = smf.ols(formula = 'wpc ~ pyth', data=engl_summary).fit()
pyth_lm.summary()


# # Questions

# Can we use Pythagorean Expectation to predict the outcome of a game?

# # Quiz

# How many EPL games were played in 2018 (from 2017-18 season)

# In[18]:


(engl_17
# .head()
 .query("Date.str.startswith('2018')")
 .query("division == 1")
 .shape
)


# Which team scored the highest number of goals while playing at home in the first half of the season?
# 
# (division 1?)

# In[34]:


engl_17.head()


# In[33]:


(
    engl_17.assign(
        half=lambda x: np.where(
            x["Date"].str.startswith("2017"),
            2017,
            np.where(x["Date"].str.startswith("2018"), 2018, 999),
        )
    )
    .query("division == 1")
    .query("half == 2017")
    .groupby('home')
    .agg({'hgoal': np.sum})
#     ['hgoal'].sum()
    .sort_values('hgoal', ascending=False)
#     .groupby('half')
)


# Which team conceded the highest number of goals while playing away in the first half of the season?

# In[35]:


(
    engl_17.assign(
        half=lambda x: np.where(
            x["Date"].str.startswith("2017"),
            2017,
            np.where(x["Date"].str.startswith("2018"), 2018, 999),
        )
    )
    .query("division == 1")
    .query("half == 2017")
    .groupby('visitor')
    .agg({'hgoal': np.sum})
    .sort_values('hgoal', ascending=False)
)


# Which of the following teams had the smallest difference between their win percentage and Pythagorean expectation in the first half of the season?

# In[42]:


engl_home_half = (
    engl_17.assign(
        half=lambda x: np.where(
            x["Date"].str.startswith("2017"),
            2017,
            np.where(x["Date"].str.startswith("2018"), 2018, 999),
        )
    )
    .groupby(["half", "home", "division"])["count", "hwin", "hgoal", "vgoal"]
    .sum()
    .reset_index()
    .rename(
        columns={
            "home": "team",
            "count": "Ph",
            "hgoal": "FTHGh",
            "vgoal": "FTAGh",
#             'hwin': 'hwinvalue'
        }
    )
)
engl_home_half


# In[43]:


engl_away_half = (
    engl_17.assign(
        half=lambda x: np.where(
            x["Date"].str.startswith("2017"),
            2017,
            np.where(x["Date"].str.startswith("2018"), 2018, 999),
        )
    )
    .groupby(["half", "visitor"])["count", "awin", "hgoal", "vgoal"]
    .sum()
    .reset_index()
    .rename(
        columns={
            "visitor": "team",
            "count": "Pa",
            "hgoal": "FTHGa",
            "vgoal": "FTAGa",
        }
    )
)
engl_away_half


# In[45]:


# GF - goals scores
# GA - goals conceded

engl_summary_half = (
    engl_home_half.merge(engl_away_half, on=["team", "half"])
    .query("division == 1")
    .assign(
        W=lambda x: x["hwin"] + x["awin"],
        G=lambda x: x["Ph"] + x["Pa"],
        GF=lambda x: x["FTHGh"] + x["FTAGa"],
        GA=lambda x: x["FTAGh"] + x["FTHGa"],
    )
    .assign(
        wpc=lambda x: x["W"] / x["G"],
        pyth=lambda x: x["GF"] ** 2 / (x["GF"] ** 2 + x["GA"] ** 2),
    )
)
engl_summary_half


# In[ ]:


engl_summary_half.query("half == 2017").assign(
    delta=lambda x: abs(x["pyth"] - x["wpc"])
).sort_values("delta")


# In[52]:


(engl_summary_half.query("half == 2018").assign(
    delta=lambda x: abs(x["hwin"] - x["awin"])
).sort_values("delta"))


# In[60]:


cols = ['team', 'wpc', 'pyth']
(engl_summary_half.query("half == 2017")[cols]
.merge(engl_summary_half.query("half == 2018")[cols],
      on=["team"], suffixes=["_1", "_2"])
.corr())


# In[ ]:




