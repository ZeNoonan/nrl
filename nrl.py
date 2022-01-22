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
finished_week=26

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
# team_names_id = (read_csv_data('C:/Users/Darragh/Documents/Python/nrl/nrl_team_id.csv')).drop(['Unnamed: 0'],axis=1).copy()
team_names_id = id_excel
# st.write(team_names_id)
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
data=data.drop(['Date','Unnamed: 0'],axis=1)
data['Date']=pd.to_datetime(data[['year','month','day']])
data['Week'] = data['Week'].replace({'Finals':26})
data['Week']=pd.to_numeric(data['Week'])
team_names_id=team_names_id.rename(columns={'Team':'Home Team'})
# st.write('original team id',team_names_id)
# st.write(data)
fb_ref_2020=pd.merge(data,team_names_id,on='Home Team').rename(columns={'ID':'Home ID'})
# st.write('after merge',fb_ref_2020)
team_names_id_2=team_names_id.rename(columns={'Home Team':'Away Team'})
# st.write('team id 2',team_names_id_2)
data=pd.merge(fb_ref_2020,team_names_id_2,on='Away Team').rename(columns={'ID':'Away ID','Home Score':'Home Points',
'Away Score':'Away Points','Home Line Close':'Spread'})
cols_to_move=['Week','Date','Home ID','Home Team','Away ID','Away Team','Spread']
cols = cols_to_move + [col for col in data if col not in cols_to_move]
data=data[cols]

def spread_workings(data):
    data['home_win']=data['Home Points'] - data['Away Points']
    data['home_win'] = np.where((data['Home Points'] > data['Away Points']), 1, np.where((data['Home Points'] < data['Away Points']),-1,0))
    data['home_cover']=(np.where(((data['Home Points'] + data['Spread']) > data['Away Points']), 1,
    np.where(((data['Home Points']+ data['Spread']) < data['Away Points']),-1,0)))
    data['home_cover']=data['home_cover'].astype(int)
    data['away_cover'] = -data['home_cover']
    data['Date']=pd.to_datetime(data['Date'])
    # data=data.rename(columns={'Net Turnover':'home_turnover'})
    # data['away_turnover'] = -data['home_turnover']
    return data

def turnover_workings(data,week_start):
    turnover_df=data[data['Week']>week_start].copy()
    turnover_df['home_turned_over_sign'] = np.where((turnover_df['Turnover'] > 0), 1, np.where((turnover_df['Turnover'] < 0),-1,0))
    turnover_df['away_turned_over_sign'] = - turnover_df['home_turned_over_sign']
    # season_cover_df=(data.set_index('Week').loc[week_start:,:]).reset_index()
    home_turnover_df = (turnover_df.loc[:,['Week','Date','Home ID','home_turned_over_sign']]).rename(columns={'Home ID':'ID','home_turned_over_sign':'turned_over_sign'})
    # st.write('checking home turnover section', home_turnover_df[home_turnover_df['ID']==0])
    away_turnover_df = (turnover_df.loc[:,['Week','Date','Away ID','away_turned_over_sign']]).rename(columns={'Away ID':'ID','away_turned_over_sign':'turned_over_sign'})
    # st.write('checking away turnover section', away_turnover_df[away_turnover_df['ID']==0])
    season_cover=pd.concat([home_turnover_df,away_turnover_df],ignore_index=True)
    # season_cover_df = pd.melt(season_cover_df,id_vars=['Week', 'home_cover'],value_vars=['Home ID', 'Away ID']).set_index('Week').rename(columns={'value':'ID'}).\
    # drop('variable',axis=1).reset_index().sort_values(by=['Week','ID'],ascending=True)
    return season_cover.sort_values(by=['Week','Date','ID'],ascending=['True','True','True'])

def turnover_2(season_cover_df):    
    # https://stackoverflow.com/questions/53335567/use-pandas-shift-within-a-group
    season_cover_df['prev_turnover']=season_cover_df.groupby('ID')['turned_over_sign'].shift()
    return season_cover_df.sort_values(by=['ID','Week'],ascending=True)
    # return season_cover_df

def season_cover_3(data,column_sign,name):
    data[column_sign] = np.where((data[name] > 0), 1, np.where((data[name] < 0),-1,0))
    return data

def penalty_workings(data,week_start):
    turnover_df=data[data['Week']>week_start].copy()
    turnover_df['home_penalty_sign'] = np.where((turnover_df['penalties_conceded'] > 0), 1, np.where((turnover_df['penalties_conceded'] < 0),-1,0))
    turnover_df['away_penalty_sign'] = - turnover_df['home_penalty_sign']
    # season_cover_df=(data.set_index('Week').loc[week_start:,:]).reset_index()
    home_turnover_df = (turnover_df.loc[:,['Week','Date','Home ID','home_penalty_sign']]).rename(columns={'Home ID':'ID','home_penalty_sign':'penalty_sign'})
    # st.write('checking home turnover section', home_turnover_df[home_turnover_df['ID']==0])
    away_turnover_df = (turnover_df.loc[:,['Week','Date','Away ID','away_penalty_sign']]).rename(columns={'Away ID':'ID','away_penalty_sign':'penalty_sign'})
    # st.write('checking away turnover section', away_turnover_df[away_turnover_df['ID']==0])
    season_cover=pd.concat([home_turnover_df,away_turnover_df],ignore_index=True)
    # season_cover_df = pd.melt(season_cover_df,id_vars=['Week', 'home_cover'],value_vars=['Home ID', 'Away ID']).set_index('Week').rename(columns={'value':'ID'}).\
    # drop('variable',axis=1).reset_index().sort_values(by=['Week','ID'],ascending=True)
    return season_cover.sort_values(by=['Week','Date','ID'],ascending=['True','True','True'])

def penalty_2(season_cover_df):    
    # https://stackoverflow.com/questions/53335567/use-pandas-shift-within-a-group
    # st.write('before line in function', season_cover_df)
    season_cover_df['prev_penalty']=season_cover_df.groupby('ID')['penalty_sign'].shift()
    # st.write('after line in function', season_cover_df)
    season_cover_df = season_cover_df.sort_values(by=['ID','Week'],ascending=True)
    # st.write('last check in function', season_cover_df)
    x = season_cover_df
    return x

def clean_version_of_above_which_works(x):
    # don't know why this version works and above doesn't
    x['prev_penalty']=x.groupby('ID')['penalty_sign'].shift()
    return x.sort_values(by=['ID','Week'],ascending=[True,True])

def penalty_cover_3(data,column_sign,name):
    data[column_sign] = np.where((data[name] > 0), 1, np.where((data[name] < 0),-1,0))
    return data

turnover=spread_workings(data)
# st.write('turnover workings', turnover)
turnover_1 = turnover_workings(turnover,-1)
turnover_2=turnover_2(turnover_1)
turnover_3=season_cover_3(turnover_2,'turnover_sign','prev_turnover')

penalty=spread_workings(data)
st.write('SPREAD WORKINGS where is penalty??', penalty)
penalty_1 = penalty_workings(penalty,-1)
penalty_2=penalty_2(penalty_1)
penalty_3=penalty_cover_3(penalty_2,'penalty_sign','prev_penalty')

# intercept_0=spread_workings(data)
# intercept_0['intercepts']=intercept_0['home_intercepts']
intercept=spread_workings(data).drop(['penalties_conceded'],axis=1).rename(columns={'intercepts':'penalties_conceded'})
# st.write(intercept)
intercept_1 = pd.DataFrame(penalty_workings(intercept,-1))
intercept_2=clean_version_of_above_which_works(intercept_1)
intercept_3=penalty_cover_3(intercept_2,'penalty_sign','prev_penalty')

sin_bin=spread_workings(data).drop(['penalties_conceded'],axis=1).rename(columns={'sin_bin':'penalties_conceded'})
sin_bin_1 = pd.DataFrame(penalty_workings(sin_bin,-1))
sin_bin_2=clean_version_of_above_which_works(sin_bin_1)
sin_bin_3=penalty_cover_3(sin_bin_2,'penalty_sign','prev_penalty')


def season_cover_workings(data,home,away,name,week_start):
    season_cover_df=data[data['Week']>week_start].copy()
    # season_cover_df=(data.set_index('Week').loc[week_start:,:]).reset_index()
    home_cover_df = (season_cover_df.loc[:,['Week','Date','Home ID',home]]).rename(columns={'Home ID':'ID',home:name})
    # st.write('checking home turnover section', home_cover_df[home_cover_df['ID']==0])
    away_cover_df = (season_cover_df.loc[:,['Week','Date','Away ID',away]]).rename(columns={'Away ID':'ID',away:name})
    # st.write('checking away turnover section', away_cover_df[away_cover_df['ID']==0])
    season_cover=pd.concat([home_cover_df,away_cover_df],ignore_index=True)
    # season_cover_df = pd.melt(season_cover_df,id_vars=['Week', 'home_cover'],value_vars=['Home ID', 'Away ID']).set_index('Week').rename(columns={'value':'ID'}).\
    # drop('variable',axis=1).reset_index().sort_values(by=['Week','ID'],ascending=True)
    return season_cover.sort_values(by=['Week','Date','ID'],ascending=['True','True','True'])

def season_cover_2(season_cover_df,column_name):    
    # https://stackoverflow.com/questions/54993050/pandas-groupby-shift-and-cumulative-sum
    # season_cover_df[column_name] = season_cover_df.groupby (['ID'])[column_name].transform(lambda x: x.cumsum().shift())
    # THE ABOVE DIDN'T WORK IN 2020 PRO FOOTBALL BUT DID WORK IN 2019 DO NOT DELETE FOR INFO PURPOSES
    season_cover_df[column_name] = season_cover_df.groupby (['ID'])[column_name].apply(lambda x: x.cumsum().shift())
    season_cover_df=season_cover_df.reset_index().sort_values(by=['Week','Date','ID'],ascending=True).drop('index',axis=1)
    # Be careful with this if you want full season, season to date cover, for week 17, it is season to date up to week 16
    # if you want full season, you have to go up to week 18 to get the full 17 weeks, just if you want to do analysis on season covers
    return season_cover_df

spread=spread_workings(data)

# with st.beta_expander('Season to date Cover'):
spread_1 = season_cover_workings(spread,'home_cover','away_cover','cover',0)
spread_2=season_cover_2(spread_1,'cover')
spread_3=season_cover_3(spread_2,'cover_sign','cover')

matrix_df=spread_workings(data)
matrix_df=matrix_df.reset_index().rename(columns={'index':'unique_match_id'})
test_df = matrix_df.copy()
matrix_df['at_home'] = 1
matrix_df['at_away'] = -1
matrix_df['home_pts_adv'] = -3
matrix_df['away_pts_adv'] = 3
matrix_df['away_spread']=-matrix_df['Spread']
matrix_df=matrix_df.rename(columns={'Spread':'home_spread'})
matrix_df_1=matrix_df.loc[:,['unique_match_id','Week','Home ID','Away ID','at_home','at_away','home_spread','away_spread','home_pts_adv','away_pts_adv','Date','Home Points','Away Points']].copy()
# st.write('checking matrix', matrix_df_1.dtypes)

# with st.beta_expander('Games Played to be used in Matrix Multiplication'):
first_qtr=matrix_df_1.copy()
start=-3
finish=0
first_4=first_qtr[first_qtr['Week'].between(start,finish)].copy()
def games_matrix_workings(first_4):
    group_week = first_4.groupby('Week')
    raw_data_2=[]
    game_weights = iter([-0.125, -0.25,-0.5,-1])
    for name, group in group_week:
        group['game_adj']=next(game_weights)
        raw_data_2.append(group)

    df3 = pd.concat(raw_data_2, ignore_index=True)
    adj_df3=df3.loc[:,['Home ID', 'Away ID', 'game_adj']].copy()
    test_adj_df3 = adj_df3.rename(columns={'Home ID':'Away ID', 'Away ID':'Home ID'})
    concat_df_test=pd.concat([adj_df3,test_adj_df3]).sort_values(by=['Home ID', 'game_adj'],ascending=[True,False])
    test_concat_df_test=concat_df_test.groupby('Home ID')['game_adj'].sum().abs().reset_index()
    test_concat_df_test['Away ID']=test_concat_df_test['Home ID']
    full=pd.concat([concat_df_test,test_concat_df_test]).sort_values(by=['Home ID', 'game_adj'],ascending=[True,False])
    full_stack=pd.pivot_table(full,index='Away ID', columns='Home ID',aggfunc='sum')
    full_stack=full_stack.fillna(0)
    full_stack.columns = full_stack.columns.droplevel(0)
    return full_stack
    
    full_stack=games_matrix_workings(first_4)
    # st.write('Check sum if True all good', full_stack.sum().sum()==0)

# with st.beta_expander('CORRECT Testing reworking the DataFrame'):
test_df['at_home'] = 1
test_df['at_away'] = -1
test_df['home_pts_adv'] = 3
test_df['away_pts_adv'] = -3
test_df['away_spread']=-test_df['Spread']
test_df=test_df.rename(columns={'Spread':'home_spread'})
test_df_1=test_df.loc[:,['unique_match_id','Week','Home ID','Away ID','at_home','at_away','home_spread','away_spread','home_pts_adv','away_pts_adv']].copy()
test_df_home=test_df_1.loc[:,['Week','Home ID','at_home','home_spread','home_pts_adv']].rename(columns={'Home ID':'ID','at_home':'home','home_spread':'spread','home_pts_adv':'home_pts_adv'}).copy()
test_df_away=test_df_1.loc[:,['Week','Away ID','at_away','away_spread','away_pts_adv']].rename(columns={'Away ID':'ID','at_away':'home','away_spread':'spread','away_pts_adv':'home_pts_adv'}).copy()
test_df_2=pd.concat([test_df_home,test_df_away],ignore_index=True)
test_df_2=test_df_2.sort_values(by=['ID','Week'],ascending=True)
test_df_2['spread_with_home_adv']=test_df_2['spread']+test_df_2['home_pts_adv']
# st.write(test_df_2)

def test_4(matrix_df_1):
    weights = np.array([0.125, 0.25,0.5,1])
    sum_weights = np.sum(weights)
    matrix_df_1['adj_spread']=matrix_df_1['spread_with_home_adv'].rolling(window=4, center=False).apply(lambda x: np.sum(weights*x), raw=False)
    return matrix_df_1

# with st.beta_expander('CORRECT Power Ranking to be used in Matrix Multiplication'):
    # # https://stackoverflow.com/questions/9621362/how-do-i-compute-a-weighted-moving-average-using-pandas
grouped = test_df_2.groupby('ID')
# https://stackoverflow.com/questions/16974047/efficient-way-to-find-missing-elements-in-an-integer-sequence
# https://stackoverflow.com/questions/62471485/is-it-possible-to-insert-missing-sequence-numbers-in-python
ranking_power=[]
for name, group in grouped:
    dfseq = pd.DataFrame.from_dict({'Week': range( -3,27 )}).merge(group, on='Week', how='outer').fillna(np.NaN)
    dfseq['ID']=dfseq['ID'].fillna(method='ffill')
    dfseq['home_pts_adv']=dfseq['home_pts_adv'].fillna(0)
    dfseq['spread']=dfseq['spread'].fillna(0)
    dfseq['spread_with_home_adv']=dfseq['spread_with_home_adv'].fillna(0)
    dfseq['home']=dfseq['home'].fillna(0)
    df_seq_1 = dfseq.groupby(['Week','ID'])['spread_with_home_adv'].sum().reset_index()
    update=test_4(df_seq_1)
    ranking_power.append(update)

    df_power = pd.concat(ranking_power, ignore_index=True)
    # st.write('power ranking',df_power.sort_values(by=['ID','Week'],ascending=[True,True]))
    # st.write('power ranking',df_power.sort_values(by=['Week','ID'],ascending=[True,True]))

# with st.beta_expander('CORRECT Power Ranking Matrix Multiplication'):
# https://stackoverflow.com/questions/62775018/matrix-array-multiplication-whats-excel-doing-mmult-and-how-to-mimic-it-in#62775508
inverse_matrix=[]
power_ranking=[]
list_inverse_matrix=[]
list_power_ranking=[]
power_df=df_power.loc[:,['Week','ID','adj_spread']].copy()
games_df=matrix_df_1.copy()
first=list(range(-3,24))
last=list(range(0,27))
for first,last in zip(first,last):
    first_section=games_df[games_df['Week'].between(first,last)]
    full_game_matrix=games_matrix_workings(first_section)
    adjusted_matrix=full_game_matrix.loc[0:14,0:14]
    df_inv = pd.DataFrame(np.linalg.pinv(adjusted_matrix.values), adjusted_matrix.columns, adjusted_matrix.index)
    power_df_week=power_df[power_df['Week']==last].drop_duplicates(subset=['ID'],keep='last').set_index('ID')\
    .drop('Week',axis=1).rename(columns={'adj_spread':0}).loc[:14,:]
    result = df_inv.dot(pd.DataFrame(power_df_week))
    result.columns=['power']
    avg=(result['power'].sum())/16
    result['avg_pwr_rank']=(result['power'].sum())/16
    result['final_power']=result['avg_pwr_rank']-result['power']
    df_pwr=pd.DataFrame(columns=['final_power'],data=[avg])
    result=pd.concat([result,df_pwr],ignore_index=True)
    result['week']=last+1
    power_ranking.append(result)

    power_ranking_combined = pd.concat(power_ranking).reset_index().rename(columns={'index':'ID'})
    # st.write('power ranking combined', power_ranking_combined)

# with st.beta_expander('Adding Power Ranking to Matches'):
matches_df = spread.copy()
home_power_rank_merge=power_ranking_combined.loc[:,['ID','week','final_power']].copy().rename(columns={'week':'Week','ID':'Home ID'})
away_power_rank_merge=power_ranking_combined.loc[:,['ID','week','final_power']].copy().rename(columns={'week':'Week','ID':'Away ID'})
updated_df=pd.merge(matches_df,home_power_rank_merge,on=['Home ID','Week']).rename(columns={'final_power':'home_power'})
updated_df=pd.merge(updated_df,away_power_rank_merge,on=['Away ID','Week']).rename(columns={'final_power':'away_power'})
updated_df['calculated_spread']=updated_df['away_power']-updated_df['home_power']
updated_df['spread_working']=updated_df['home_power']-updated_df['away_power']+updated_df['Spread']
updated_df['power_pick'] = np.where(updated_df['spread_working'] > 0, 1,
np.where(updated_df['spread_working'] < 0,-1,0))
# updated_df_1=updated_df.copy()
# st.write(updated_df)

with st.expander('Season to Date Cover Factor by Team'):
    st.write('Positive number means the number of games to date that you have covered the spread; in other words teams with a positive number have beaten expectations')
    st.write('Negative number means the number of games to date that you have not covered the spread; in other words teams with a negative number have performed below expectations')
    st.write('blanks in graph are where the team got a bye week')
    stdc_home=spread_3.rename(columns={'ID':'Home ID'})
    stdc_home['cover_sign']=-stdc_home['cover_sign']
    stdc_away=spread_3.rename(columns={'ID':'Away ID'})
    updated_df=updated_df.drop(['away_cover'],axis=1)
    updated_df=updated_df.rename(columns={'home_cover':'home_cover_result'})
    updated_df=updated_df.merge(stdc_home,on=['Date','Week','Home ID'],how='left').rename(columns={'cover':'home_cover','cover_sign':'home_cover_sign'})
    updated_df=pd.merge(updated_df,stdc_away,on=['Date','Week','Away ID'],how='left').rename(columns={'cover':'away_cover','cover_sign':'away_cover_sign'})
    updated_df_1=updated_df.copy()
    
    stdc_df=pd.merge(spread_3,team_names_id,on='ID').rename(columns={'Home Team':'Team'})
    stdc_df=stdc_df.loc[:,['Week','Team','cover']].copy()
    stdc_df['average']=stdc_df.groupby('Team')['cover'].transform(np.mean)
    # st.write(stdc_df.sort_values(by=['Team','Week']))
    stdc_pivot=pd.pivot_table(stdc_df,index='Team', columns='Week')
    stdc_pivot.columns = stdc_pivot.columns.droplevel(0)
    chart_cover= alt.Chart(stdc_df).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    alt.Y('Team',sort=alt.SortField(field='average', order='descending')),color=alt.Color('cover:Q',scale=alt.Scale(scheme='redyellowgreen')))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    text_cover=chart_cover.mark_text().encode(text=alt.Text('cover:N'),color=alt.value('black'))
    st.altair_chart(chart_cover + text_cover,use_container_width=True)

with st.expander('Turnover Factor by Match Graph'):
    st.write('-1 means you received more turnovers than other team, 1 means you gave up more turnovers to other team')

    intercept_3=intercept_3.rename(columns={'penalty_sign':'turnover_sign', 'prev_penalty':'prev_turnover'})
    sin_bin_3=sin_bin_3.rename(columns={'penalty_sign':'turnover_sign', 'prev_penalty':'prev_turnover'})

    def turnover_data_prep_1(turnover_3):
        return turnover_3.loc[:,['Date','Week','ID','prev_turnover', 'turnover_sign']].copy()

    turnover_matches = turnover_data_prep_1(turnover_3)
    intercept_matches = turnover_data_prep_1(intercept_3)
    sin_bin_matches = turnover_data_prep_1(sin_bin_3)
    # st.write('turnover matches',turnover_matches)
    
    # st.write(turnover_matches['Date'].dtypes)
    # st.write(turnover_matches['ID'].dtypes)
    # st.write(turnover_matches['Week'].dtypes)
    # st.write(turnover_matches['prev_turnover'].dtypes)
    # st.write(turnover_matches['turnover_sign'].dtypes)
    # st.write('intercept matches',intercept_matches)

    # st.write(intercept_matches['Date'].dtypes)
    # st.write(intercept_matches['ID'].dtypes)
    # st.write(intercept_matches['Week'].dtypes)
    # st.write(intercept_matches['prev_turnover'].dtypes)
    # st.write(intercept_matches['turnover_sign'].dtypes)


    def turnover_data_prep_2(turnover_matches,updated_df):
        turnover_home=turnover_matches.rename(columns={'ID':'Home ID'})
        turnover_away=turnover_matches.rename(columns={'ID':'Away ID'})
        turnover_away['turnover_sign']=-turnover_away['turnover_sign']
        updated_df=pd.merge(updated_df,turnover_home,on=['Date','Week','Home ID'],how='left').rename(columns={'prev_turnover':'home_prev_turnover','turnover_sign':'home_turnover_sign'})
        updated_df=pd.merge(updated_df,turnover_away,on=['Date','Week','Away ID'],how='left').rename(columns={'prev_turnover':'away_prev_turnover','turnover_sign':'away_turnover_sign'})
        return updated_df

    updated_df = turnover_data_prep_2(turnover_matches, updated_df)
    updated_df_intercept = turnover_data_prep_2(intercept_matches, updated_df_1)
    updated_df_sin_bin = turnover_data_prep_2(sin_bin_matches, updated_df_1)
    # st.write(updated_df_intercept)
    # st.write(updated_df_intercept.dtypes)

    df_stdc_1=pd.merge(turnover_matches,team_names_id,on='ID').rename(columns={'Home Team':'Team'})
    # st.write(df_stdc_1)
    df_stdc_1['average']=df_stdc_1.groupby('Team')['turnover_sign'].transform(np.mean)

    color_scale = alt.Scale(domain=[1,0,-1],range=["red", "lightgrey","LimeGreen"])

    chart_cover= alt.Chart(df_stdc_1).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    # alt.Y('Team',sort=alt.SortField(field='average', order='ascending')),color=alt.Color('turnover_sign:Q',scale=alt.Scale(scheme='redyellowgreen')))
    alt.Y('Team',sort=alt.SortField(field='average', order='ascending')),color=alt.Color('turnover_sign:Q',scale=color_scale))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    
    text_cover=chart_cover.mark_text().encode(text=alt.Text('turnover_sign:N'),color=alt.value('black'))
    st.altair_chart(chart_cover + text_cover,use_container_width=True)

with st.expander('Penalty Factor by Match Graph'):
    st.write('-1 means you received more turnovers than other team, 1 means you gave up more turnovers to other team')
    # st.write('this is turnovers', turnover_3)
    def run_penalty_workings_1(penalty_3):
        return penalty_3.loc[:,['Date','Week','ID','prev_penalty', 'penalty_sign']].copy()
    penalty_matches=run_penalty_workings_1(penalty_3)
    
    penalty_home=penalty_matches.rename(columns={'ID':'Home ID'})
    penalty_away=penalty_matches.rename(columns={'ID':'Away ID'})
    penalty_away['penalty_sign']=-penalty_away['penalty_sign']
    penalty_df=pd.merge(updated_df,penalty_home,on=['Date','Week','Home ID'],how='left').rename(columns={'prev_penalty':'home_prev_penalty',
    'penalty_sign':'home_penalty_sign'})

    penalty_df=pd.merge(penalty_df,penalty_away,on=['Date','Week','Away ID'],how='left').rename(columns={'prev_penalty':'away_prev_penalty',
    'penalty_sign':'away_penalty_sign'})

    df_stdc_2=pd.merge(penalty_matches,team_names_id,on='ID').rename(columns={'Home Team':'Team'})
    # st.write(df_stdc_1)
    df_stdc_2['average']=df_stdc_2.groupby('Team')['penalty_sign'].transform(np.mean)

    color_scale = alt.Scale(domain=[1,0,-1],range=["red", "lightgrey","LimeGreen"])

    chart_cover= alt.Chart(df_stdc_2).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    # alt.Y('Team',sort=alt.SortField(field='average', order='ascending')),color=alt.Color('turnover_sign:Q',scale=alt.Scale(scheme='redyellowgreen')))
    alt.Y('Team',sort=alt.SortField(field='average', order='ascending')),color=alt.Color('penalty_sign:Q',scale=color_scale))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    
    text_cover=chart_cover.mark_text().encode(text=alt.Text('penalty_sign:N'),color=alt.value('black'))
    st.altair_chart(chart_cover + text_cover,use_container_width=True)
    # st.write(updated_df)
    penalty_df=penalty_df.drop(['home_turnover_sign','away_turnover_sign'],axis=1)
    penalty_df=penalty_df.rename(columns={'home_penalty_sign':'home_turnover_sign','away_penalty_sign':'away_turnover_sign'})
    # st.write(updated_df)
    # updated_df=penalty_df

with st.expander('Betting Slip Matches'):
    def run_analysis(updated_df):
        betting_matches=updated_df.loc[:,['Week','Date','Home ID','Home Team','Away ID', 'Away Team','Spread','Home Points','Away Points',
        'home_power','away_power','home_cover','away_cover','home_turnover_sign','away_turnover_sign',
        'home_cover_sign','away_cover_sign','power_pick','home_cover_result']]
        betting_matches['total_factor']=betting_matches['home_turnover_sign']+betting_matches['away_turnover_sign']+betting_matches['home_cover_sign']+\
        betting_matches['away_cover_sign']+betting_matches['power_pick']
        betting_matches['bet_on'] = np.where(betting_matches['total_factor']>2,betting_matches['Home Team'],np.where(betting_matches['total_factor']<-2,betting_matches['Away Team'],''))
        betting_matches['bet_sign'] = (np.where(betting_matches['total_factor']>2,1,np.where(betting_matches['total_factor']<-2,-1,0)))
        betting_matches['bet_sign'] = betting_matches['bet_sign'].astype(float)
        betting_matches['home_cover'] = betting_matches['home_cover'].astype(float)
        betting_matches['result']=betting_matches['home_cover_result'] * betting_matches['bet_sign']
        st.write('testing sum of betting result',betting_matches['result'].sum())
        # this is for graphing anlaysis on spreadsheet
        betting_matches['bet_sign_all'] = (np.where(betting_matches['total_factor']>0,1,np.where(betting_matches['total_factor']<-0,-1,0)))
        betting_matches['result_all']=betting_matches['home_cover_result'] * betting_matches['bet_sign_all']
        st.write('testing sum of betting all result',betting_matches['result_all'].sum())
        cols_to_move=['Week','Date','Home Team','Away Team','total_factor','bet_on','result','Spread','Home Points','Away Points','home_power','away_power']
        cols = cols_to_move + [col for col in betting_matches if col not in cols_to_move]
        betting_matches=betting_matches[cols]
        betting_matches=betting_matches.sort_values(['Week','Date'],ascending=[True,True])
        return betting_matches
    # st.write(betting_matches.dtypes)
    # st.write(betting_matches)
    betting_matches_penalty=run_analysis(penalty_df)
    betting_matches_intercept=run_analysis(updated_df_intercept)
    betting_matches_sin_bin=run_analysis(updated_df_sin_bin)
    betting_matches=run_analysis(updated_df)
    presentation_betting_matches=betting_matches.copy()

    # https://towardsdatascience.com/7-reasons-why-you-should-use-the-streamlit-aggrid-component-2d9a2b6e32f0
    grid_height = st.number_input("Grid height", min_value=400, value=550, step=100)
    gb = GridOptionsBuilder.from_dataframe(presentation_betting_matches)
    gb.configure_column("Spread", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=1, aggFunc='sum')
    gb.configure_column("home_power", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=1, aggFunc='sum')
    gb.configure_column("away_power", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=1, aggFunc='sum')
    gb.configure_column("Date", type=["dateColumnFilter","customDateTimeFormat"], custom_format_string='dd-MM-yyyy', pivot=True)
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)



    test_cellsytle_jscode = JsCode("""
    function(params) {
        if (params.value < 0) {
        return {
            'color': 'red',
        }
        } else {
            return {
                'color': 'black',
            }
        }
    };
    """)
    # # https://github.com/PablocFonseca/streamlit-aggrid/blob/main/st_aggrid/grid_options_builder.py
    gb.configure_column(field="Spread", cellStyle=test_cellsytle_jscode)
    gb.configure_column("home_power", cellStyle=test_cellsytle_jscode)
    gb.configure_column("away_power", cellStyle=test_cellsytle_jscode)


    # gb.configure_pagination()
    # gb.configure_side_bar()
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()
    grid_response = AgGrid(
        presentation_betting_matches, 
        gridOptions=gridOptions,
        height=grid_height, 
        width='100%',
        # data_return_mode=return_mode_value, 
        # update_mode=update_mode_value,
        # fit_columns_on_grid_load=fit_columns_on_grid_load,
        allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
        enable_enterprise_modules=True,
    )

    # container.grid_response
    # AgGrid(betting_matches.sort_values('Date').style.format({'home_power':"{:.1f}",'away_power':"{:.1f}"}))
    # update




    st.write('Below is just checking an individual team')
    betting_matches_team=betting_matches.copy()
    # cols_to_move_now=['Week','Home Team','Away Team','Spread','Home Points','Away Points','home_cover_result','home_cover','away_cover']
    # cols = cols_to_move_now + [col for col in betting_matches_team if col not in cols_to_move_now]
    # betting_matches_team=betting_matches_team[cols]
    # st.write( betting_matches_team[(betting_matches_team['Home Team']=='Melbourne Storm') | 
    # (betting_matches_team['Away Team']=='Melbourne Storm')].sort_values(by='Week').set_index('Week') )

with st.expander('Power Pick Factor by Team'):
    st.write('Positive number means the market has undervalued the team as compared to the spread')
    st.write('Negative number means the market has overvalued the team as compared to the spread')    
    power_factor=betting_matches.loc[:,['Week','Home Team','Away Team','power_pick']].rename(columns={'power_pick':'home_power_pick'})
    power_factor['away_power_pick']=-power_factor['home_power_pick']
    home_factor=power_factor.loc[:,['Week','Home Team','home_power_pick']].rename(columns={'Home Team':'Team','home_power_pick':'power_pick'})
    away_factor=power_factor.loc[:,['Week','Away Team','away_power_pick']].rename(columns={'Away Team':'Team','away_power_pick':'power_pick'})
    graph_power_pick=pd.concat([home_factor,away_factor],axis=0).sort_values(by=['Week'])
    graph_power_pick['average']=graph_power_pick.groupby('Team')['power_pick'].transform(np.mean)

    color_scale = alt.Scale(domain=[1,0,-1],range=["LimeGreen", "lightgrey","red"])

    chart_cover= alt.Chart(graph_power_pick).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    # alt.Y('Team',sort=alt.SortField(field='average', order='ascending')),color=alt.Color('turnover_sign:Q',scale=alt.Scale(scheme='redyellowgreen')))
    alt.Y('Team',sort=alt.SortField(field='average', order='descending')),color=alt.Color('power_pick:Q',scale=color_scale))
    text_cover=chart_cover.mark_text().encode(text=alt.Text('power_pick:N'),color=alt.value('black'))
    st.altair_chart(chart_cover + text_cover,use_container_width=True)
    # st.write('graph',graph_power_pick)
    # st.write('data',power_factor)



with st.expander('Power Ranking by Week'):
    power_week=power_ranking_combined.copy()
    team_names_id=team_names_id.rename(columns={'Home Team':'Team'})
    id_names=team_names_id.drop_duplicates(subset=['ID'], keep='first')
    pivot_df=pd.merge(power_week,id_names, on='ID')
    pivot_df=pivot_df.loc[:,['Team','final_power','week']].copy()
    power_pivot=pd.pivot_table(pivot_df,index='Team', columns='week')
    pivot_df_test = pivot_df.copy()
    pivot_df_test=pivot_df_test[pivot_df_test['week']<finished_week+1]
    pivot_df_test['average']=pivot_df.groupby('Team')['final_power'].transform(np.mean)
    power_pivot.columns = power_pivot.columns.droplevel(0)
    power_pivot['average'] = power_pivot.mean(axis=1)
    # https://stackoverflow.com/questions/67045668/altair-text-over-a-heatmap-in-a-script
    pivot_df=pivot_df.sort_values(by='final_power',ascending=False)
    chart_power= alt.Chart(pivot_df_test).mark_rect().encode(alt.X('week:O',axis=alt.Axis(title='week',labelAngle=0)),
    alt.Y('Team',sort=alt.SortField(field='average', order='descending')),color=alt.Color('final_power:Q',scale=alt.Scale(scheme='redyellowgreen')))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    text=chart_power.mark_text().encode(text=alt.Text('final_power:N',format=",.0f"),color=alt.value('black'))
    st.altair_chart(chart_power + text,use_container_width=True)
    # https://github.com/altair-viz/altair/issues/820#issuecomment-386856394

with st.expander('Analysis of Betting Results across 1 to 5 factors'):
    # matches_in_regular_season= (32 * 16) / 2
    # st.write('In 2020 there were 13 matches in playoffs looks like this was new so 269 total matches in 2020 season compared with 267 in previous seasons')
    # matches_in_playoffs = 13
    # total_matches =matches_in_regular_season + matches_in_playoffs
    # st.write('total_matches per my calculation',total_matches)
    analysis=betting_matches.copy()
    # totals = analysis.groupby('total_factor').agg(winning=('result_all','count'))
    # totals_1=analysis.groupby([analysis['total_factor'].abs(),'result_all']).agg(winning=('result_all','count')).reset_index()
    # totals_1['result_all']=totals_1['result_all'].replace({0:'tie',1:'win',-1:'lose'})

    # st.write('shows the number of games at each factor level')
    # st.write(totals.rename(columns={'winning':'number_of_games'}))
    # st.write('sum of each factor level should correspond to table above',totals_1)
    # st.write('sum of winning column should be 267 I think',totals_1['winning'].sum())
    # st.write('count of week column should be 267',analysis['Week'].count())

    analysis=analysis[analysis['Week']<finished_week+1]
    # st.write('analysis',analysis)
    totals = analysis.groupby('total_factor').agg(winning=('result_all','count'))
    totals_graph=totals.reset_index().rename(columns={'winning':'number_of_games'})

    # st.write('totals grpah', totals_graph)
    chart_power= alt.Chart(totals_graph).mark_bar().encode(alt.X('total_factor:O',axis=alt.Axis(title='total_factor_per_match',labelAngle=0)),
    alt.Y('number_of_games'))
    text=chart_power.mark_text(dy=-7).encode(text=alt.Text('number_of_games:N',format=",.0f"),color=alt.value('black'))
    st.altair_chart(chart_power + text,use_container_width=True)
    
    totals_1=analysis.groupby([analysis['total_factor'].abs(),'result_all']).agg(winning=('result_all','count')).reset_index()
    totals_1['result_all']=totals_1['result_all'].replace({0:'tie',1:'win',-1:'lose'})
    # st.write('checking graph data',totals_1.dtypes)
    # totals_1['total_factor']=totals_1['total_factor'].astype(str)
    # st.write('checking graph data',totals_1.dtypes)
    # st.write('checking graph data',totals_1)
    # https://www.quackit.com/css/css_color_codes.cfm
    color_scale = alt.Scale(
    domain=[
        "lose",
        "tie",
        "win"],
        range=["red", "lightgrey","LimeGreen"])
    chart_power= alt.Chart(totals_1).mark_bar().encode(alt.X('total_factor:O',axis=alt.Axis(title='factor',labelAngle=0)),
    alt.Y('winning'),color=alt.Color('result_all',scale=color_scale))
    # st.altair_chart(chart_power,use_container_width=True)

    
    normalized_table = (totals_1[totals_1['result_all']!='tie']).copy()
    # st.write('graph date to be cleaned',totals_1)
    chart_power= alt.Chart(normalized_table).mark_bar().encode(alt.X('total_factor:O',axis=alt.Axis(title='factor',labelAngle=0)),
    alt.Y('winning',stack="normalize"),color=alt.Color('result_all',scale=color_scale))
    overlay = pd.DataFrame({'winning': [0.5]})
    vline = alt.Chart(overlay).mark_rule(color='black', strokeWidth=2).encode(y='winning:Q')
    text = alt.Chart(normalized_table).mark_text(dx=-1, dy=+60, color='white').encode(
    x=alt.X('total_factor:O'),
    y=alt.Y('winning',stack="normalize"),
    detail='winning',
    text=alt.Text('winning:Q', format='.0f'))
    updated_test_chart=chart_power+vline+text
    
    st.altair_chart(updated_test_chart,use_container_width=True)

    # st.write('shows the number of games at each factor level')
    # st.write(totals.rename(columns={'winning':'number_of_games'}))
    # st.write('sum of each factor level should correspond to table above',totals_1)
    # st.write('sum of winning column should be 267 I think',totals_1['winning'].sum())
    # st.write('count of week column should be 267',analysis['Week'].count())

with st.expander('Analysis of Factors'):
    def run_data(betting_matches):
        analysis_factors = betting_matches.copy()
        analysis_factors=analysis_factors[analysis_factors['Week']<finished_week+1]
        return analysis_factors

    analysis_factors=run_data(betting_matches)
    analysis_factors_penalty=run_data(betting_matches_penalty)
    analysis_factors_intercept=run_data(betting_matches_intercept)
    analysis_factors_sin_bin=run_data(betting_matches_sin_bin)

    # st.write('check for penalties', analysis_factors)
    def analysis_factor_function(analysis_factors,option_1='home_turnover_sign',option_2='away_turnover_sign'):
        analysis_factors.loc[:,['home_turnover_success?']] = analysis_factors['home_turnover_sign'] * analysis_factors['home_cover_result']
        analysis_factors.loc[:,['away_turnover_success?']] = analysis_factors['away_turnover_sign'] * analysis_factors['home_cover_result']
        analysis_factors.loc[:,['home_cover_season_success?']] = analysis_factors['home_cover_sign'] * analysis_factors['home_cover_result']  
        analysis_factors.loc[:,['away_cover_season_success?']] = analysis_factors['away_cover_sign'] * analysis_factors['home_cover_result']
        analysis_factors.loc[:,['power_ranking_success?']] = analysis_factors['power_pick'] * analysis_factors['home_cover_result']
        df_table = analysis_factors['home_turnover_success?'].value_counts()
        away_turnover=analysis_factors['away_turnover_success?'].value_counts()
        home_cover=analysis_factors['home_cover_season_success?'].value_counts()
        away_cover=analysis_factors['away_cover_season_success?'].value_counts()
        power=analysis_factors['power_ranking_success?'].value_counts()
        df_table_1=pd.concat([df_table,away_turnover,home_cover,away_cover,power],axis=1)
        # df_table_1=pd.concat([df_table,away_turnover,home_cover,away_cover,power],axis=1).reset_index().drop('index',axis=1)
        # st.write('df table', df_table_1)
        # test=df_table_1.reset_index()
        # st.write(test)
        df_table_1['total_turnover'] = df_table_1['home_turnover_success?'].add (df_table_1['away_turnover_success?'])
        # st.write(test)
        df_table_1['total_season_cover'] = df_table_1['home_cover_season_success?'] + df_table_1['away_cover_season_success?']
        
        df_table_1=df_table_1.reset_index()
        df_table_1['index']=df_table_1['index'].astype(str)
        df_table_1=df_table_1.set_index('index')
        # st.write('df table 2', df_table_1)
        df_table_1.loc['Total']=df_table_1.sum()
        # st.write('latest', df_table_1)
        # st.write('latest', df_table_1.shape)
        if df_table_1.shape > (2,7):
            # st.write('Returning df with analysis')
            # df_table_1.loc['No. of Bets Made'] = df_table_1.loc[[1,-1]].sum() # No losing bets so far!!!
            df_table_1.loc['No. of Bets Made'] = df_table_1.loc[['1','-1']].sum() # No losing bets so far!!!
            # df_table_1.loc['% Winning'] = ((df_table_1.loc[1] / df_table_1.loc['No. of Bets Made'])*100).apply('{:,.1f}%'.format)
            df_table_1.loc['% Winning'] = ((df_table_1.loc['1'] / df_table_1.loc['No. of Bets Made']))
        else:
            # st.write('Returning df with no analysis')
            return df_table_1
        return df_table_1
    total_factor_table = analysis_factor_function(analysis_factors)
    total_factor_table_penalty = analysis_factor_function(analysis_factors_penalty)
    total_factor_table_intercept = analysis_factor_function(analysis_factors_intercept)
    total_factor_table_sin_bin = analysis_factor_function(analysis_factors_sin_bin)

    
    
    def clean_presentation_table(total_factor_table):
        total_factor_table=total_factor_table.loc[:,['total_turnover','total_season_cover','power_ranking_success?']]
        # st.write(total_factor_table.dtypes)
        reorder_list=['1','-1','0','Total','No. of Bets Made','% Winning']
        total_factor_table=total_factor_table.reindex(reorder_list)
        total_factor_table_presentation = total_factor_table.style.format("{:.0f}", na_rep='-')
        total_factor_table_presentation = total_factor_table_presentation.format(formatter="{:.1%}", subset=pd.IndexSlice[['% Winning'], :])
        return total_factor_table_presentation

    total_factor_table_presentation=clean_presentation_table(total_factor_table)
    total_factor_table_presentation_penalty=clean_presentation_table(total_factor_table_penalty)
    total_factor_table_presentation_intercept=clean_presentation_table(total_factor_table_intercept)
    total_factor_table_presentation_sin_bin=clean_presentation_table(total_factor_table_sin_bin)
    columns_1,columns_2=st.columns(2)
    columns_3,columns_4=st.columns(2)

    def betting_df(analysis_factors):
        return analysis_factors[analysis_factors['bet_sign']!=0]

    factor_bets=betting_df(analysis_factors)
    factor_bets_penalty=betting_df(analysis_factors_penalty)
    factor_bets_intercept=betting_df(analysis_factors_intercept)
    factor_bets_sin_bin=betting_df(analysis_factors_sin_bin)
    # factor_bets = (analysis_factors[analysis_factors['bet_sign']!=0]).copy()
    bets_made_factor_table = analysis_factor_function(factor_bets)
    bets_made_factor_table_presentation=clean_presentation_table(bets_made_factor_table)
    bets_made_factor_table_penalty = analysis_factor_function(factor_bets_penalty)
    bets_made_factor_table_intercept = analysis_factor_function(factor_bets_intercept)
    bets_made_factor_table_sin_bin = analysis_factor_function(factor_bets_sin_bin)
    bets_made_factor_table_presentation_penalty=clean_presentation_table(bets_made_factor_table_penalty)
    bets_made_factor_table_presentation_intercept=clean_presentation_table(bets_made_factor_table_intercept)
    bets_made_factor_table_presentation_sin_bin=clean_presentation_table(bets_made_factor_table_sin_bin)

    with columns_1:
        st.subheader('This represents the Error factor')
        st.write('This is the total number of matches broken down by Factor result')
        st.write(total_factor_table_presentation)
        st.write('This is the number of bets made broken down by Factor result')
        st.write(bets_made_factor_table_presentation)

    with columns_2:
        st.subheader('This represents the Penalty factor')
        st.write('This is the total number of matches broken down by Factor result')
        st.write(total_factor_table_presentation_penalty)
        st.write('This is the number of bets made broken down by Factor result')
        st.write(bets_made_factor_table_presentation_penalty)

    with columns_3:
        st.subheader('This represents the Intercept factor')
        st.write('This is the total number of matches broken down by Factor result')
        st.write(total_factor_table_presentation_intercept)
        st.write('This is the number of bets made broken down by Factor result')
        st.write(bets_made_factor_table_presentation_intercept)

    with columns_4:
        st.subheader('This represents the sin_bin factor')
        st.write('This is the total number of matches broken down by Factor result')
        st.write(total_factor_table_presentation_sin_bin)
        st.write('This is the number of bets made broken down by Factor result')
        st.write(bets_made_factor_table_presentation_sin_bin)

    graph_factor_table = total_factor_table.copy().loc[['-1','0','1'],:].reset_index().rename(columns={'index':'result_all'})
    graph_factor_table['result_all']=graph_factor_table['result_all'].replace({'0':'tie','1':'win','-1':'lose'})
    graph_factor_table=graph_factor_table.melt(id_vars='result_all',var_name='total_factor',value_name='winning')
    chart_power= alt.Chart(graph_factor_table).mark_bar().encode(alt.X('total_factor:O',axis=alt.Axis(title='factor',labelAngle=0)),
    alt.Y('winning'),color=alt.Color('result_all',scale=color_scale))
    # alt.Y('winning'),color=alt.Color('result_all'))
    # st.write('do the normalised stacked bar chart which shows percentage')
    # st.altair_chart(chart_power,use_container_width=True)

    normalized_table = graph_factor_table.copy()
    normalized_table=normalized_table[normalized_table['result_all']!='tie']
    normalized_table= normalized_table[(normalized_table['total_factor']=='total_turnover') | (normalized_table['total_factor']=='total_season_cover')
     | (normalized_table['total_factor']=='power_ranking_success?')].copy()
    chart_power= alt.Chart(normalized_table).mark_bar().encode(alt.X('total_factor:O',axis=alt.Axis(title='factor',labelAngle=0)),
    alt.Y('winning',stack="normalize"),color=alt.Color('result_all',scale=color_scale))
    overlay = pd.DataFrame({'winning': [0.5]})
    vline = alt.Chart(overlay).mark_rule(color='black', strokeWidth=2).encode(y='winning:Q')
    
    
    text = alt.Chart(normalized_table).mark_text(dx=-1, dy=+37, color='white').encode(
    x=alt.X('total_factor:O'),
    y=alt.Y('winning',stack="normalize"),
    detail='winning',
    text=alt.Text('winning:Q', format='.0f'))
    
    # chart_power=chart_power+text

    # updated_test_chart = alt.layer(chart_power,vline)
    updated_test_chart=chart_power+vline
    # updated_test_chart=chart_power+vline+text
    
    st.altair_chart(updated_test_chart,use_container_width=True)

    reset_data=totals_1.copy()
    # reset_data['result_all']=reset_data['result_all'].replace({'tie':0,'win':1,'lose':-1})
    reset_data['result_all']=reset_data['result_all'].replace({'tie':'0','win':'1','lose':'-1'})
    reset_data=reset_data.pivot(index='result_all',columns='total_factor',values='winning').fillna(0)
    reset_data['betting_factor_total']=reset_data[3]+reset_data[4]+reset_data[5]
    reset_data=reset_data.sort_values(by='betting_factor_total',ascending=False)
    # st.write('working???',reset_data)
    reset_data.loc['Total']=reset_data.sum()
    # reset_data.loc['No. of Bets Made'] = reset_data.loc[[1,-1]].sum()
    reset_data.loc['No. of Bets Made'] = reset_data.loc[['1','-1']].sum()
    reset_data=reset_data.apply(pd.to_numeric, downcast='integer')
    # reset_data.loc['% Winning'] = ((reset_data.loc[1] / reset_data.loc['No. of Bets Made'])*100).apply('{:,.1f}%'.format)
    reset_data.loc['% Winning'] = ((reset_data.loc['1'] / reset_data.loc['No. of Bets Made']))
    reorder_list=['1','-1','0','Total','No. of Bets Made','% Winning']
    reset_data=reset_data.reindex(reorder_list)

    st.write('This shows the betting result')
    reset_data = reset_data.style.format("{:.0f}", na_rep='-')
    reset_data = reset_data.format(formatter="{:.1%}", subset=pd.IndexSlice[['% Winning'], :])

    st.write(reset_data)
    st.write('Broken down by the number of factors indicating the strength of the signal')

# with st.expander('Analysis of Penalty Factors'):
#     analysis_factors = betting_matches.copy()
#     analysis_factors=analysis_factors[analysis_factors['Week']<finished_week+1]
#     # st.write('check for penalties', analysis_factors)
#     def analysis_factor_function(analysis_factors):
#         analysis_factors['home_penalty_success?'] = analysis_factors['home_penalty_sign'] * analysis_factors['home_cover_result']
#         analysis_factors['away_penalty_success?'] = analysis_factors['away_penalty_sign'] * analysis_factors['home_cover_result']
#         analysis_factors['home_cover_season_success?'] = analysis_factors['home_cover_sign'] * analysis_factors['home_cover_result']  
#         analysis_factors['away_cover_season_success?'] = analysis_factors['away_cover_sign'] * analysis_factors['home_cover_result']
#         analysis_factors['power_ranking_success?'] = analysis_factors['power_pick'] * analysis_factors['home_cover_result']
#         # df_table = analysis_factors['home_penalty_success?'].value_counts()
#         # away_turnover=analysis_factors['away_penalty_success?'].value_counts()
#         home_cover=analysis_factors['home_cover_season_success?'].value_counts()
#         away_cover=analysis_factors['away_cover_season_success?'].value_counts()
#         power=analysis_factors['power_ranking_success?'].value_counts()
#         df_table_1=pd.concat([df_table,away_turnover,home_cover,away_cover,power],axis=1)
#         # df_table_1=pd.concat([df_table,away_turnover,home_cover,away_cover,power],axis=1).reset_index().drop('index',axis=1)
#         # st.write('df table', df_table_1)
#         # test=df_table_1.reset_index()
#         # st.write(test)
#         df_table_1['total_penalty'] = df_table_1['home_penalty_success?'].add (df_table_1['away_penalty_success?'])
#         # st.write(test)
#         df_table_1['total_season_cover'] = df_table_1['home_cover_season_success?'] + df_table_1['away_cover_season_success?']
#         # st.write('df table 2', df_table_1)
#         df_table_1=df_table_1.reset_index()
#         df_table_1['index']=df_table_1['index'].astype(str)
#         df_table_1=df_table_1.set_index('index')
#         df_table_1.loc['Total']=df_table_1.sum()
#         # st.write('latest', df_table_1)
#         # st.write('latest', df_table_1.shape)
#         if df_table_1.shape > (2,7):
#             # st.write('Returning df with analysis')
#             # df_table_1.loc['No. of Bets Made'] = df_table_1.loc[[1,-1]].sum() # No losing bets so far!!!
#             df_table_1.loc['No. of Bets Made'] = df_table_1.loc[['1','-1']].sum() # No losing bets so far!!!
#             df_table_1.loc['% Winning'] = ((df_table_1.loc['1'] / df_table_1.loc['No. of Bets Made'])*100)
#         else:
#             # st.write('Returning df with no analysis')
#             return df_table_1
#         return df_table_1
#     total_factor_table = analysis_factor_function(analysis_factors)   
#     st.write('This is the total number of matches broken down by Factor result')
#     cols_to_move=['total_penalty','total_season_cover','power_ranking_success?']
#     total_factor_table = total_factor_table[ cols_to_move + [ col for col in total_factor_table if col not in cols_to_move ] ]
#     total_factor_table=total_factor_table.loc[:,['total_penalty','total_season_cover','power_ranking_success?']]
#     st.write(total_factor_table)
#     factor_bets = (analysis_factors[analysis_factors['bet_sign_penalty']!=0]).copy()
#     bets_made_factor_table = analysis_factor_function(factor_bets)
#     # cols_to_move=['total_turnover','total_season_cover','power_ranking_success?']
#     bets_made_factor_table = bets_made_factor_table[ cols_to_move + [ col for col in bets_made_factor_table if col not in cols_to_move ] ]
#     bets_made_factor_table=bets_made_factor_table.loc[:,['total_penalty','total_season_cover','power_ranking_success?']]
#     st.write('This is the matches BET ON broken down by Factor result')
#     st.write(bets_made_factor_table)

#     # st.write('graph work below')
#     # graph_factor_table = total_factor_table.copy().loc[[-1,0,1],:].reset_index().rename(columns={'index':'result_all_penalty'})
#     graph_factor_table = total_factor_table.copy().loc[['-1','0','1'],:].reset_index().rename(columns={'index':'result_all_penalty'})
#     # graph_factor_table['result_all_penalty']=graph_factor_table['result_all_penalty'].replace({0:'tie',1:'win',-1:'lose'})
#     graph_factor_table['result_all_penalty']=graph_factor_table['result_all_penalty'].replace({'0':'tie','1':'win','-1':'lose'})
#     graph_factor_table=graph_factor_table.melt(id_vars='result_all_penalty',var_name='total_factor_penalty',value_name='winning')
#     chart_power= alt.Chart(graph_factor_table).mark_bar().encode(alt.X('total_factor_penalty:O',axis=alt.Axis(title='factor',labelAngle=0)),
#     alt.Y('winning'),color=alt.Color('result_all_penalty',scale=color_scale))
#     # alt.Y('winning'),color=alt.Color('result_all'))
#     # st.write('do the normalised stacked bar chart which shows percentage')
#     # st.altair_chart(chart_power,use_container_width=True)

#     normalized_table = graph_factor_table.copy()
#     normalized_table=normalized_table[normalized_table['result_all_penalty']!='tie']
#     # st.write('normalised table 1', normalized_table)
#     normalized_table= normalized_table[(normalized_table['total_factor_penalty']=='total_penalty') | (normalized_table['total_factor_penalty']=='total_season_cover')
#      | (normalized_table['total_factor_penalty']=='power_ranking_success?')].copy()
#     # st.write('normalised table 2', normalized_table) 
#     chart_power= alt.Chart(normalized_table).mark_bar().encode(alt.X('total_factor_penalty:O',axis=alt.Axis(title='factor',labelAngle=0)),
#     alt.Y('winning',stack="normalize"),color=alt.Color('result_all_penalty',scale=color_scale))
#     overlay = pd.DataFrame({'winning': [0.5]})
#     vline = alt.Chart(overlay).mark_rule(color='black', strokeWidth=2).encode(y='winning:Q')
    
    
#     text = alt.Chart(normalized_table).mark_text(dx=-1, dy=+57, color='white').encode(
#     x=alt.X('total_factor_penalty:O'),
#     y=alt.Y('winning',stack="normalize"),
#     detail='winning',
#     text=alt.Text('winning:Q', format='.0f'))
    
#     # chart_power=chart_power+text

#     # updated_test_chart = alt.layer(chart_power,vline)
#     updated_test_chart=chart_power+vline+text
    
#     st.altair_chart(updated_test_chart,use_container_width=True)

with st.expander('Checking Performance where Total Factor = 2 or 3:  Additional Diagnostic'):
    df_factor = betting_matches.copy()
    two_factor_df = df_factor[df_factor['total_factor'].abs()==2]
    # st.write(two_factor_df)
    factor_2_3_home_turnover_filter = (df_factor['total_factor']==2)&(df_factor['home_turnover_sign']==-1) | \
    (df_factor['total_factor']==-2)&(df_factor['home_turnover_sign']==1) | (df_factor['total_factor']==3)&(df_factor['home_turnover_sign']==1) | \
    (df_factor['total_factor']==-3)&(df_factor['home_turnover_sign']==-1)

    factor_2_3_away_turnover_filter = (df_factor['total_factor']==2)&(df_factor['away_turnover_sign']==-1) | \
    (df_factor['total_factor']==-2)&(df_factor['away_turnover_sign']==1) | (df_factor['total_factor']==3)&(df_factor['away_turnover_sign']==1) | \
    (df_factor['total_factor']==-3)&(df_factor['away_turnover_sign']==-1)

    factor_2_3_home_cover_filter = (df_factor['total_factor']==2)&(df_factor['home_cover_sign']==-1) | \
    (df_factor['total_factor']==-2)&(df_factor['home_cover_sign']==1) | (df_factor['total_factor']==3)&(df_factor['home_cover_sign']==1) | \
    (df_factor['total_factor']==-3)&(df_factor['home_cover_sign']==-1)

    factor_2_3_away_cover_filter = (df_factor['total_factor']==2)&(df_factor['away_cover_sign']==-1) | \
    (df_factor['total_factor']==-2)&(df_factor['away_cover_sign']==1) | (df_factor['total_factor']==3)&(df_factor['away_cover_sign']==1) | \
    (df_factor['total_factor']==-3)&(df_factor['away_cover_sign']==-1)

    factor_2_3_power_filter = (df_factor['total_factor']==2)&(df_factor['power_pick']==-1) | \
    (df_factor['total_factor']==-2)&(df_factor['power_pick']==1) | (df_factor['total_factor']==3)&(df_factor['power_pick']==1) | \
    (df_factor['total_factor']==-3)&(df_factor['power_pick']==-1)

    df_factor['home_turnover_diagnostic'] = (df_factor['home_turnover_sign'].where(factor_2_3_home_turnover_filter)) * df_factor['home_cover_result']
    df_factor['away_turnover_diagnostic'] = (df_factor['away_turnover_sign'].where(factor_2_3_away_turnover_filter)) * df_factor['home_cover_result']
    df_factor['home_cover_diagnostic'] = (df_factor['home_cover_sign'].where(factor_2_3_home_cover_filter)) * df_factor['home_cover_result']
    df_factor['away_cover_diagnostic'] = (df_factor['away_cover_sign'].where(factor_2_3_away_cover_filter)) * df_factor['home_cover_result']
    df_factor['power_diagnostic'] = (df_factor['power_pick'].where(factor_2_3_power_filter)) * df_factor['home_cover_result']
    # st.write(df_factor)

    df_factor_table = df_factor['home_turnover_diagnostic'].value_counts()
    away_turnover=df_factor['away_turnover_diagnostic'].value_counts()
    home_cover=df_factor['home_cover_diagnostic'].value_counts()
    away_cover=df_factor['away_cover_diagnostic'].value_counts()
    power=df_factor['power_diagnostic'].value_counts()
    df_factor_table_1=pd.concat([df_factor_table,away_turnover,home_cover,away_cover,power],axis=1)
    df_factor_table_1['total_turnover'] = df_factor_table_1['home_turnover_diagnostic'].add (df_factor_table_1['away_turnover_diagnostic'])
    # st.write(test)
    df_factor_table_1['total_season_cover'] = df_factor_table_1['home_cover_diagnostic'] + df_factor_table_1['away_cover_diagnostic']
    # st.write('df table 2', df_factor_table_1)

    df_factor_table_1=df_factor_table_1.reset_index()
    df_factor_table_1['index']=df_factor_table_1['index'].astype(int)
    # st.write('df table 2', df_factor_table_1)
    df_factor_table_1['index']=df_factor_table_1['index'].astype(str)
    # st.write('df table 2', df_factor_table_1)
    df_factor_table_1=df_factor_table_1.set_index('index')
    # st.write('df table 2', df_factor_table_1)
    df_factor_table_1.loc['Total']=df_factor_table_1.sum()
    # st.write('latest', df_factor_table_1)
    # st.write('latest', df_factor_table_1.shape)

    if df_factor_table_1.shape > (2,7):
        df_factor_table_1.loc['No. of Bets Made'] = df_factor_table_1.loc[['1','-1']].sum() 
        df_factor_table_1.loc['% Winning'] = ((df_factor_table_1.loc['1'] / df_factor_table_1.loc['No. of Bets Made']))
    # else:
    #     # st.write('Returning df with no anal')
    #     return df_factor_table_1


    cols_to_move=['total_turnover','total_season_cover','power_diagnostic']
    df_factor_table_1 = df_factor_table_1[ cols_to_move + [ col for col in df_factor_table_1 if col not in cols_to_move ] ]
    df_factor_table_1=df_factor_table_1.loc[:,['total_turnover','total_season_cover','power_diagnostic']]
    df_factor_table_1_presentation = df_factor_table_1.style.format("{:.0f}", na_rep='-')
    df_factor_table_1_presentation = df_factor_table_1_presentation.format(formatter="{:.1%}", subset=pd.IndexSlice[['% Winning'], :])

    st.write(df_factor_table_1_presentation)