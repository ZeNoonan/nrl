import pandas as pd
import numpy as np
import streamlit as st
# from io import BytesIO
# import os
# import base64 
import altair as alt
import datetime as dt
# from st_aggrid import AgGrid
from st_aggrid import AgGrid, GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode

st.set_page_config(layout="wide")

results_excel=pd.read_excel('C:/Users/Darragh/Documents/Python/nrl/nrl.xlsx')
id_excel=pd.read_excel('C:/Users/Darragh/Documents/Python/nrl/nrl_id.xlsx')

def csv_save(x):
    x.to_csv('C:/Users/Darragh/Documents/Python/nrl/nrl_data.csv')
    return x
# csv_save(results_excel)

@st.cache
def read_csv_data(file):
    return pd.read_csv(file)

@st.cache
def read_csv_data_date(file):
    return pd.read_csv(file,parse_dates=['Date'])

# url = read_csv_data('https://raw.githubusercontent.com/ZeNoonan/nrl/main/nrl_data.csv').copy()
url = 'https://raw.githubusercontent.com/ZeNoonan/nrl/main/nrl_data.csv'
# https://www.aussportsbetting.com/data/historical-nfl-results-and-odds-data/
# team_names_id = read_csv_data('https://raw.githubusercontent.com/ZeNoonan/nrl/main/nrl_team_id.csv').copy()
team_names_id = read_csv_data('C:/Users/Darragh/Documents/Python/nrl/nrl_data.csv').copy()

# st.write(pd.read_csv(url))
# data=pd.read_csv(url,parse_dates=['Date'])
local='C:/Users/Darragh/Documents/Python/nrl/nrl_data.csv'
# data=pd.read_csv(local,parse_dates=['Date'])
data=(read_csv_data_date(local)).copy()
# data=pd.read_csv(url,parse_dates=['Date'])
# st.write(data)

# data['Date']=pd.to_datetime(data['Date'],errors='coerce')
data['year']=data['Date'].dt.year
data['month']=data['Date'].dt.month
data['day']=data['Date'].dt.day
data=data.drop(['Date'],axis=1)
data['Date']=pd.to_datetime(data[['year','month','day']])

team_names_id=team_names_id.rename(columns={'Team':'Home Team'})
st.write(team_names_id)
st.write(data)
fb_ref_2020=pd.merge(data,team_names_id,on='Home Team').rename(columns={'ID':'Home ID'})
# st.write(fb_ref_2020)
team_names_id_2=team_names_id.rename(columns={'Home Team':'Away Team'})
data=pd.merge(fb_ref_2020,team_names_id_2,on='Away Team').rename(columns={'ID':'Away ID','Home Score':'Home Points',
'Away Score':'Away Points','Home Line Close':'Spread'})
cols_to_move=['Week','Date','Home ID','Home Team','Away ID','Away Team','Spread']
cols = cols_to_move + [col for col in data if col not in cols_to_move]
data=data[cols]

st.write(data)