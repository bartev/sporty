#!/usr/bin/env python
# coding: utf-8

# # Week 1

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


# # Read data

# In[3]:


pwd


# In[4]:


with open('../data/raw/wk1-baseball/retrosheet-gamelog-header.txt', 'r') as f:
    retro_cols = f.read().split(',')


# In[5]:


len(retro_cols)


# In[6]:


MLB = pd.read_csv('../data/raw/wk1-baseball/GL2018.csv', names=retro_cols)
MLB


# In[15]:


MLB18 = (MLB
        [['VisitingTeam', 'HomeTeam', 'VisitorRunsScored','HomeRunsScore', 'Date']]
        .rename(columns={'VisitorRunsScored': 'VisR', 'HomeRunsScore': 'HomeR'}))
MLB18


# In[16]:


# Assign home/away win

MLB18 = MLB18.assign(
    hwin=lambda x: np.where(x["HomeR"] > x["VisR"], 1, 0),
    awin=lambda x: np.where(x["HomeR"] < x["VisR"], 1, 0),
    count=1,
)


# In[17]:


MLB18


# In[18]:


MLBhome = (
    MLB18.groupby("HomeTeam")["hwin", "HomeR", "VisR", "count"]
    .sum()
    .reset_index()
    .rename(
        columns={"HomeTeam": "team", "VisR": "VisRh", "HomeR": "HomeRh", "count": "Gh"}
    )
)
MLBhome


# In[19]:


# VisRa = vistiting team runs scored away

MLBaway = (
    MLB18.groupby("VisitingTeam")["awin", "HomeR", "VisR", "count"]
    .sum()
    .reset_index()
    .rename(
        columns={"VisitingTeam": "team", "VisR": "VisRa", "HomeR": "HomeRa", "count": "Ga"}
    )
)
MLBaway


# In[25]:


# HomeRh = runs scored as a home team
# VisRh = runs scored by other team when we're home
# HomeRa = runs scored by other team when we're away
# VisRa = runs scored by us when we're away
MLB18 = pd.merge(MLBhome, MLBaway, on='team')
MLB18


# In[26]:


MLB18.columns


# wpc = the win percentage
# pyth = the pythagorean expectation
# 
# $$
# \text{pythagorean expectation} = \frac{\text{runs scored}^2}{\text{runs scored}^2 + \text{runs allowd}^2}
# $$

# In[35]:


MLB18 = MLB18.assign(
    W=lambda x: x["hwin"] + x["awin"],
    G=lambda x: x["Gh"] + x["Ga"],
    R=lambda x: x["HomeRh"] + x["VisRa"],
    RA=lambda x: x["VisRh"] + x["HomeRa"],
).assign(
    wpc=lambda x: x["W"] / x["G"],
    pyth=lambda x: x["R"] ** 2 / (x["R"] ** 2 + x["RA"] ** 2),
)


# # Draw a picture

# In[36]:


MLB18.head()


# In[64]:


# relplot is a scatterplot
# Illustrates the close correlation between win percentage and pythagorean expectation

sns.relplot(x='pyth', y='wpc', data=MLB18)


# # Generate a regression using `statsmodel`

# https://www.statsmodels.org/stable/index.html
# 
# $$
# \text{wpc} = \text{intercept} + \text{coef} \times \text{pyth}
# $$

# In[58]:


# ols = ordinary least squares
# using R style formulas
python_lm = smf.ols(formula='wpc ~ pyth', data=MLB18).fit()


# In[59]:


results = python_lm.summary()
results


# Or, using numpy/pandas
# 
# Don't forget to add an intercept column!

# In[50]:


import statsmodels.api as sm


# In[62]:


results2 = sm.OLS(MLB18['wpc'], MLB18[['pyth']].assign(intercept=1)).fit()
results2.summary()


# In[45]:


type(results)


# In[44]:


results


# In[40]:


print(python_lm.summary())


# Interpretation
# 
# * coef = 0.8770  (slope of the curve)
# * t-statistic: coef / std_err
#     * Tells us about statistical significance (see p-value)
# * P-value (P > t) is the probability that we'd observe the value (0.877) by chance if the true value was 0.
#     * By convention, if P-value > 0.05, we are NOT confident that our value is not 0.
# * R^2: Tells us the percentage of the variation in `wpc` which can be accounted for by the variation in `pyth`. Here, the pythagorean expectation accounts for 89.4% of the variation in win percentage

# In[46]:


.877 / 0.057


# In[ ]:




