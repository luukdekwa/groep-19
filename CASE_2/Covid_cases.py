#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install kaggle


# - full_grouped.csv - Day to day country wise no. of cases (Has County/State/Province level data)
# - covid19clean_complete.csv - Day to day country wise no. of cases (Doesn't have County/State/Province level data)
# - countrywiselatest.csv - Latest country level no. of cases
# - day_wise.csv - Day wise no. of cases (Doesn't have country level data)
# - usacountywise.csv - Day to day county level no. of cases
# - worldometer_data.csv - Latest data from https://www.worldometers.info/


#!mkdir C:\Users\luuk\.kaggle

#!kaggle datasets download -d sudalairajkumar/novel-corona-virus-2019-dataset


# In[5]:


#import zipfile
#with zipfile.ZipFile('novel-corona-virus-2019-dataset.zip', 'r') as zip_ref:
#    zip_ref.extractall()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot,plot
import plotly.express as px
import json
import datetime
import seaborn as sns

# Nog 4 csv's, kunnen alvast wat onderzoek naar doen en wat cleaning gebeuren


df = pd.read_csv('"../groep-19/blob/main/time_series_covid_19_confirmed.csv"',index_col='Country/Region')
df.drop(['Province/State','Lat','Long'], axis=1, inplace=True)
#df.dropna(axis=1, thresh=1, subset="Province/State")
print(df)
list(df)


dfLong = df.T
print(dfLong)

df1 = pd.read_csv('covid_19_data.csv',index_col='ObservationDate')
print(df1.head(50))
df2 = pd.read_csv('time_series_covid_19_confirmed.csv') 
# Vergeef het ons! Voor deze list (geen tijd om het op te schonen)
df2.drop(['1/22/20',
 '1/23/20',
 '1/24/20',
 '1/25/20',
 '1/26/20',
 '1/27/20',
 '1/28/20',
 '1/29/20',
 '1/30/20',
 '1/31/20',
 '2/1/20',
 '2/2/20',
 '2/3/20',
 '2/4/20',
 '2/5/20',
 '2/6/20',
 '2/7/20',
 '2/8/20',
 '2/9/20',
 '2/10/20',
 '2/11/20',
 '2/12/20',
 '2/13/20',
 '2/14/20',
 '2/15/20',
 '2/16/20',
 '2/17/20',
 '2/18/20',
 '2/19/20',
 '2/20/20',
 '2/21/20',
 '2/22/20',
 '2/23/20',
 '2/24/20',
 '2/25/20',
 '2/26/20',
 '2/27/20',
 '2/28/20',
 '2/29/20',
 '3/1/20',
 '3/2/20',
 '3/3/20',
 '3/4/20',
 '3/5/20',
 '3/6/20',
 '3/7/20',
 '3/8/20',
 '3/9/20',
 '3/10/20',
 '3/11/20',
 '3/12/20',
 '3/13/20',
 '3/14/20',
 '3/15/20',
 '3/16/20',
 '3/17/20',
 '3/18/20',
 '3/19/20',
 '3/20/20',
 '3/21/20',
 '3/22/20',
 '3/23/20',
 '3/24/20',
 '3/25/20',
 '3/26/20',
 '3/27/20',
 '3/28/20',
 '3/29/20',
 '3/30/20',
 '3/31/20',
 '4/1/20',
 '4/2/20',
 '4/3/20',
 '4/4/20',
 '4/5/20',
 '4/6/20',
 '4/7/20',
 '4/8/20',
 '4/9/20',
 '4/10/20',
 '4/11/20',
 '4/12/20',
 '4/13/20',
 '4/14/20',
 '4/15/20',
 '4/16/20',
 '4/17/20',
 '4/18/20',
 '4/19/20',
 '4/20/20',
 '4/21/20',
 '4/22/20',
 '4/23/20',
 '4/24/20',
 '4/25/20',
 '4/26/20',
 '4/27/20',
 '4/28/20',
 '4/29/20',
 '4/30/20',
 '5/1/20',
 '5/2/20',
 '5/3/20',
 '5/4/20',
 '5/5/20',
 '5/6/20',
 '5/7/20',
 '5/8/20',
 '5/9/20',
 '5/10/20',
 '5/11/20',
 '5/12/20',
 '5/13/20',
 '5/14/20',
 '5/15/20',
 '5/16/20',
 '5/17/20',
 '5/18/20',
 '5/19/20',
 '5/20/20',
 '5/21/20',
 '5/22/20',
 '5/23/20',
 '5/24/20',
 '5/25/20',
 '5/26/20',
 '5/27/20',
 '5/28/20',
 '5/29/20',
 '5/30/20',
 '5/31/20',
 '6/1/20',
 '6/2/20',
 '6/3/20',
 '6/4/20',
 '6/5/20',
 '6/6/20',
 '6/7/20',
 '6/8/20',
 '6/9/20',
 '6/10/20',
 '6/11/20',
 '6/12/20',
 '6/13/20',
 '6/14/20',
 '6/15/20',
 '6/16/20',
 '6/17/20',
 '6/18/20',
 '6/19/20',
 '6/20/20',
 '6/21/20',
 '6/22/20',
 '6/23/20',
 '6/24/20',
 '6/25/20',
 '6/26/20',
 '6/27/20',
 '6/28/20',
 '6/29/20',
 '6/30/20',
 '7/1/20',
 '7/2/20',
 '7/3/20',
 '7/4/20',
 '7/5/20',
 '7/6/20',
 '7/7/20',
 '7/8/20',
 '7/9/20',
 '7/10/20',
 '7/11/20',
 '7/12/20',
 '7/13/20',
 '7/14/20',
 '7/15/20',
 '7/16/20',
 '7/17/20',
 '7/18/20',
 '7/19/20',
 '7/20/20',
 '7/21/20',
 '7/22/20',
 '7/23/20',
 '7/24/20',
 '7/25/20',
 '7/26/20',
 '7/27/20',
 '7/28/20',
 '7/29/20',
 '7/30/20',
 '7/31/20',
 '8/1/20',
 '8/2/20',
 '8/3/20',
 '8/4/20',
 '8/5/20',
 '8/6/20',
 '8/7/20',
 '8/8/20',
 '8/9/20',
 '8/10/20',
 '8/11/20',
 '8/12/20',
 '8/13/20',
 '8/14/20',
 '8/15/20',
 '8/16/20',
 '8/17/20',
 '8/18/20',
 '8/19/20',
 '8/20/20',
 '8/21/20',
 '8/22/20',
 '8/23/20',
 '8/24/20',
 '8/25/20',
 '8/26/20',
 '8/27/20',
 '8/28/20',
 '8/29/20',
 '8/30/20',
 '8/31/20',
 '9/1/20',
 '9/2/20',
 '9/3/20',
 '9/4/20',
 '9/5/20',
 '9/6/20',
 '9/7/20',
 '9/8/20',
 '9/9/20',
 '9/10/20',
 '9/11/20',
 '9/12/20',
 '9/13/20',
 '9/14/20',
 '9/15/20',
 '9/16/20',
 '9/17/20',
 '9/18/20',
 '9/19/20',
 '9/20/20',
 '9/21/20',
 '9/22/20',
 '9/23/20',
 '9/24/20',
 '9/25/20',
 '9/26/20',
 '9/27/20',
 '9/28/20',
 '9/29/20',
 '9/30/20',
 '10/1/20',
 '10/2/20',
 '10/3/20',
 '10/4/20',
 '10/5/20',
 '10/6/20',
 '10/7/20',
 '10/8/20',
 '10/9/20',
 '10/10/20',
 '10/11/20',
 '10/12/20',
 '10/13/20',
 '10/14/20',
 '10/15/20',
 '10/16/20',
 '10/17/20',
 '10/18/20',
 '10/19/20',
 '10/20/20',
 '10/21/20',
 '10/22/20',
 '10/23/20',
 '10/24/20',
 '10/25/20',
 '10/26/20',
 '10/27/20',
 '10/28/20',
 '10/29/20',
 '10/30/20',
 '10/31/20',
 '11/1/20',
 '11/2/20',
 '11/3/20',
 '11/4/20',
 '11/5/20',
 '11/6/20',
 '11/7/20',
 '11/8/20',
 '11/9/20',
 '11/10/20',
 '11/11/20',
 '11/12/20',
 '11/13/20',
 '11/14/20',
 '11/15/20',
 '11/16/20',
 '11/17/20',
 '11/18/20',
 '11/19/20',
 '11/20/20',
 '11/21/20',
 '11/22/20',
 '11/23/20',
 '11/24/20',
 '11/25/20',
 '11/26/20',
 '11/27/20',
 '11/28/20',
 '11/29/20',
 '11/30/20',
 '12/1/20',
 '12/2/20',
 '12/3/20',
 '12/4/20',
 '12/5/20',
 '12/6/20',
 '12/7/20',
 '12/8/20',
 '12/9/20',
 '12/10/20',
 '12/11/20',
 '12/12/20',
 '12/13/20',
 '12/14/20',
 '12/15/20',
 '12/16/20',
 '12/17/20',
 '12/18/20',
 '12/19/20',
 '12/20/20',
 '12/21/20',
 '12/22/20',
 '12/23/20',
 '12/24/20',
 '12/25/20',
 '12/26/20',
 '12/27/20',
 '12/28/20',
 '12/29/20',
 '12/30/20',
 '12/31/20'], axis=1, inplace=True)
df3= pd.read_csv('covid_19_data.csv')
st.title('COVID CONFIRMED CASES VS CONFIRMED DEATHS')

test1=df3['Country/Region'].value_counts()
test1['Mainland China'] = test1['China']

df2.drop(['Lat','Long'], axis=1, inplace=True)
df3['ObservationDate'] = pd.to_datetime(df3['ObservationDate'], format = '%m/%d/%Y')
df3['Deaths'] = df3['Deaths'].astype(int)

df_2021 = df3.loc[df3['ObservationDate'].dt.year == 2021]

death_per_country = pd.DataFrame(df_2021.groupby('Country/Region')['Deaths'].max())
death_per_country.reset_index(inplace=True)
print(death_per_country.loc[death_per_country['Country/Region'] ==  'Mainland China'])

InputLand = st.sidebar.selectbox("Select Country One", (test1.index))
LandSelect = df2[df2['Country/Region'] == InputLand]
InputLand2 = st.sidebar.selectbox("Select Country Two", (test1.index))

fig = death_per_country[death_per_country['Country/Region'] == InputLand].plot(kind='bar').figure
LandSelect2 = df3[df3['Country/Region'] == InputLand]

slider = st.slider('Vergroot het plaatje', min_value=100, max_value=1000, value=500, step=50)
data = dict(type = 'choropleth',
            colorscale = 'reds',
            locations = death_per_country['Country/Region'],
            locationmode = "country names",
            z = death_per_country['Deaths'],
            text = death_per_country['Country/Region'],
            colorbar = {'title' : 'Deaths'},
        )
layout = dict(title = 'Deaths per country (2021)',
              autosize=False,
              width=slider,
              height=slider,
              geo = dict(projection = {'type':'mercator'}))
choromap = go.Figure(data = [data],layout = layout)


df4 = pd.read_csv('covid_19_data.csv')

confirmed = df4.groupby(['Country/Region'])[['Confirmed','Deaths']].max()
pd.set_option('display.max_rows', None)
confirmed.reset_index(inplace = True)

confirmed = confirmed.loc[confirmed['Deaths'] > 100]
confirmed.reset_index(inplace=True, drop=True)

us_Deaths = confirmed['Deaths'].loc[confirmed['Country/Region'] == 'US'].sum()
us_Confirmed = confirmed['Confirmed'].loc[confirmed['Country/Region'] == 'US'].sum()

world_Deaths = confirmed['Deaths'].loc[confirmed['Country/Region'] !='US'].mean()
world_confirmed = confirmed['Confirmed'].loc[confirmed['Country/Region'] != 'US'].mean()

df_US = pd.DataFrame([us_Confirmed, us_Deaths, world_confirmed, world_Deaths], index=['US_Confirmed','US_Deaths','world_Confirmed', 'world_Deaths'])
df_US.sort_index(axis=1, ascending=False, inplace=True)
ax = df_US.plot(kind='bar',figsize = (10,10), title='Total people Confirmed or died in the US', ylabel='total people (milion)').figure
#ax.set_ylabel('total people(miloen)')

#Gedeelte Luuk
df_covid_confirmed = pd.read_csv('time_series_covid_19_confirmed.csv',index_col='Country/Region')
df_covid_confirmed.drop(['Province/State', 'Lat','Long'], axis=1, inplace=True)
df_covid_confirmed = df_covid_confirmed.T
df_covid_confirmed["Date"] = df_covid_confirmed.index
df_covid_confirmed["Date"] = pd.to_datetime(df_covid_confirmed['Date'], format = '%m/%d/%y')
df_covid_confirmed = df_covid_confirmed.loc[df_covid_confirmed['Date'].dt.year == 2021]
df_covid_confirmed.index = df_covid_confirmed["Date"]
del df_covid_confirmed['Date']
df_covid_confirmed = df_covid_confirmed.T
df_covid_confirmed['total_confirmed'] = df_covid_confirmed.iloc[:, -1]
df_covid_confirmed.reset_index(inplace=True)
df_covid_max_confirmed = df_covid_confirmed[['Country/Region', 'total_confirmed']]
df_covid_max_confirmed = df_covid_max_confirmed.groupby('Country/Region')['total_confirmed'].sum().reset_index()
#df_covid_max_confirmed


# In[280]:


# Nog 4 csv's, kunnen alvast wat onderzoek naar doen en wat cleaning gebeuren
df_covid_deaths = pd.read_csv('time_series_covid_19_deaths.csv',index_col='Country/Region')
df_covid_deaths.drop(['Province/State', 'Lat','Long'], axis=1, inplace=True)
df_covid_deaths = df_covid_deaths.T

df_covid_deaths["Date"] = df_covid_deaths.index
df_covid_deaths["Date"] = pd.to_datetime(df_covid_deaths['Date'], format = '%m/%d/%y')
df_covid_deaths = df_covid_deaths.loc[df_covid_deaths['Date'].dt.year == 2021]

df_covid_deaths.index = df_covid_deaths["Date"]
del df_covid_deaths['Date']

df_covid_deaths = df_covid_deaths.T
df_covid_deaths['total_deaths'] = df_covid_deaths.iloc[:, -1]
df_covid_deaths.reset_index(inplace=True)

df_covid_max_deaths = df_covid_deaths[['Country/Region', 'total_deaths']]
df_covid_max_deaths = df_covid_max_deaths.groupby('Country/Region')['total_deaths'].sum().reset_index()

df_confimed_and_deaths = pd.merge(df_covid_max_confirmed, df_covid_max_deaths, on='Country/Region', how='left')

country_1 = df_confimed_and_deaths[df_confimed_and_deaths["Country/Region"] == InputLand]
country_2 = df_confimed_and_deaths[df_confimed_and_deaths["Country/Region"] == InputLand2]
countries = pd.concat([country_1, country_2])

fig = plt.figure(figsize=(12,12))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

ax1.set_ylabel('total confirmed', color='red')
ax2.set_ylabel('total deaths', color='blue')

width=0.2

countries=countries.set_index('Country/Region')

countries['total_confirmed'].plot(kind='bar',
                                  color='red',
                                  ax=ax1,
                                  width=width, 
                                  position=1,
                                 )
countries['total_deaths'].plot(kind='bar',
                               color='blue',
                               ax=ax2,
                               width=width, 
                               position=0)
fig.legend()
plt.title('Bar plot difference between confirmed and deathly covid cases')
plt.show()


st.plotly_chart(choromap, use_container_width=False, sharing="streamlit")
check = st.checkbox('Wil je de DataFrames laten zien?')
if check:
    st.dataframe(countries)
st.pyplot(fig=fig)
st.pyplot(fig=ax)
