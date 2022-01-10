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

# results_excel=pd.read_excel('C:/Users/Darragh/Documents/Python/nrl/nrl.xlsx')
# id_excel=pd.read_excel('C:/Users/Darragh/Documents/Python/nrl/nrl_id.xlsx')

def csv_save(x):
    x.to_csv('C:/Users/Darragh/Documents/Python/nrl/nrl_team_id.csv')
    return x
# csv_save(id_excel)

@st.cache
def read_csv_data(file):
    return pd.read_csv(file)

url = read_csv_data('https://raw.githubusercontent.com/ZeNoonan/nrl/main/nrl_data.csv').copy()
# https://www.aussportsbetting.com/data/historical-nfl-results-and-odds-data/
team_names_id = read_csv_data('https://raw.githubusercontent.com/ZeNoonan/nrl/main/nrl_team_id.csv').copy()

data=pd.read_csv(url,parse_dates=['Date'])

# data['Date']=pd.to_datetime(data['Date'],errors='coerce')
data['year']=data['Date'].dt.year
data['month']=data['Date'].dt.month
data['day']=data['Date'].dt.day
data=data.drop(['Date'],axis=1)
data['Date']=pd.to_datetime(data[['year','month','day']])

