#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install kaggle


# - full_grouped.csv - Day to day country wise no. of cases (Has County/State/Province level data)
# - covid19clean_complete.csv - Day to day country wise no. of cases (Doesn't have County/State/Province level data)
# - countrywiselatest.csv - Latest country level no. of cases
# - day_wise.csv - Day wise no. of cases (Doesn't have country level data)
# - usacountywise.csv - Day to day county level no. of cases
# - worldometer_data.csv - Latest data from https://www.worldometers.info/

# In[ ]:


#!mkdir C:\Users\luuk\.kaggle


# In[ ]:


#!kaggle datasets list -s 'fraud detection'


# In[1]:


#!kaggle datasets download -d sudalairajkumar/novel-corona-virus-2019-dataset


# In[3]:


#import zipfile
#with zipfile.ZipFile('novel-corona-virus-2019-dataset.zip', 'r') as zip_ref:
#    zip_ref.extractall()


# In[ ]:


# CODE Hierboven is eenmalig om alle csv te downloaden via kaggle API 


# In[50]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Nog 4 csv's, kunnen alvast wat onderzoek naar doen en wat cleaning gebeuren
df = pd.read_csv('time_series_covid_19_confirmed.csv',index_col='Country/Region')
df.drop(['Province/State','Lat','Long'], axis=1, inplace=True)
#df.dropna(axis=1, thresh=1, subset="Province/State")
print(df)
list(df)


# In[44]:


dfLong = df.T
print(dfLong)


# In[47]:


df.plot()
plt.show()


# In[53]:


df1 = pd.read_csv('covid_19_data.csv',index_col='SNo')
df1.head(50)


# In[ ]:


st.title('TEST')


# In[ ]:


InputDatum = st.sidebar.selectbox("Select Datum", ('01/22/2020','01/23/2020','01/24/2020','01/25/2020'))
LandSelect = df[df['ObservationDate'] == InputDatum]
st.dataframe(LandSelect)


# In[ ]:





# In[ ]:




