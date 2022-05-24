#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:32:46 2022

@author: giulia
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain, tee
import numbers
import matplotlib as mpl
import matplotlib.dates as mdates
import datetime as dt
#mpl.style.use('fivethirtyeight')
def plot_nas(df: pd.DataFrame):
    if df.isnull().sum().sum() != 0:
        print(len(df))
        print(df.isnull().sum())
        na_df = (df.isnull().sum() / len(df)) * 100      
        na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio %' :na_df})
        missing_data.plot(kind = "barh",color='#99004C')
        plt.show()
    else:
        print('No NAs found')

chains_df = pd.read_csv('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/Pricing Case Study/2. Data (CSV)/Chains.csv',na_values=[' -   '])
cons_df   = pd.read_csv('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/Pricing Case Study/2. Data (CSV)/Consolidated Financial-Shopper.csv',na_values=[' -   '])
#fin_df    = pd.read_csv('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/Pricing Case Study/2. Data (CSV)/Financial.csv',na_values=[' -   '])
#shop_df   = pd.read_csv('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/Pricing Case Study/2. Data (CSV)/Shopper.csv',na_values=[' -   '])

cons_df = cons_df.rename(columns={' PROFIT ':'Profit',' REVENUE ':'Revenue',' RoomNight ':'RoomN',' Shopper ':'Shopper','BEGIN USE DATE':'Begin Date'})

## type for each column
cons_df.dtypes
## total N of raws
#print(cons_df.shape[0]) #total raw: 282635
print(chains_df.shape[0]) #total raw = 79

## sum of null values for Profit
null_sum_prof = cons_df['Profit'].isna().sum()
print(null_sum_prof) #1156
## sum of null values for Profit
null_sum_rev = cons_df['Revenue'].isna().sum()
print(null_sum_rev) #139
## cleaning of values: converting to float
cons_df['Profit'] = cons_df['Profit'].str.replace(',','')
cons_df['Profit'] = pd.to_numeric(cons_df['Profit'])
cons_df['Revenue'] = cons_df['Revenue'].str.replace(',','')
cons_df['Revenue'] = pd.to_numeric(cons_df['Revenue'])
cons_df['Begin Date'] = pd.to_datetime(cons_df['Begin Date'])
cons_df['Book Date'] = pd.to_datetime(cons_df['Book Date'])
## verify
cons_df['Profit'].apply(type)
cons_df['Revenue'].apply(type)
cons_df['Begin Date'].apply(type)
cons_df['Book Date'].apply(type)
cons_df.dtypes

### grouping the main df by markets
market_gr = cons_df.groupby('Market (Destination) ID')
#print(market_gr.groups.keys()) #[95602, 95612, 95656]

### create df for each market
ch_subdf = market_gr.get_group(95612)
lv_subdf = market_gr.get_group(95602)
ma_subdf = market_gr.get_group(95656)
#print(ch_subdf.shape[0]) #raw = 60084
#print(lv_subdf.shape[0]) #raw = 132390
#print(ma_subdf.shape[0]) #raw = 90161

ch_subdf.dtypes

##check nan values
print(ch_subdf['Profit'].isna().sum()) #705
print(lv_subdf['Profit'].isna().sum()) #160
print(ma_subdf['Profit'].isna().sum()) #291
print(ch_subdf['Revenue'].isna().sum()) #31
print(lv_subdf['Revenue'].isna().sum()) #82
print(ma_subdf['Revenue'].isna().sum()) #26

##plot nan values

plt.figure()
plot_nas(ch_subdf)
plt.figure()
plot_nas(lv_subdf)
plt.figure()
plot_nas(ma_subdf)


list(ch_subdf.columns)
#['Profit',
# 'Revenue',
# 'RoomN',
# 'BEGIN USE DATE',
# 'Book Date',
# 'Booking Window Days',
# 'Booking Window Range',
# 'Booking Year',
# 'Booking_ID',
# 'Chain ID',
# 'Country of Demand (Site Country)',
# 'Country of Demand (Site Country) ID',
# 'Market (Destination) ID',
# 'Market (Destination) Name',
# 'ROW',
# 'Shopper',
# 'Shopper Record Number']

## Chicago
plt.figure()
ch_subdf['Rev-profit'] = ch_subdf['Revenue']-ch_subdf['Profit']
ch_subdf.plot(x='Booking Window Days',y='Rev-profit',kind='scatter',color='#00CC66')
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/ch_cost.png')
ch_gr = ch_subdf.groupby('Booking Year')
ch2015_df = ch_gr.get_group(2015)
ch2016_df = ch_gr.get_group(2016)
## Las Vegas
plt.figure()
lv_subdf['Rev-profit'] = lv_subdf['Revenue']-lv_subdf['Profit']
lv_subdf.plot(x='Booking Window Days',y='Rev-profit',kind='scatter',color='#00CC66')
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/lv_cost.png')
lv_gr = lv_subdf.groupby('Booking Year')
lv2015_df = lv_gr.get_group(2015)
lv2016_df = lv_gr.get_group(2016)
## Manhattan
plt.figure()
ma_subdf['Rev-profit'] = ma_subdf['Revenue']-ma_subdf['Profit']
ax = ma_subdf.plot(x='Booking Window Days',y='Rev-profit',kind='scatter',color='#00CC66')
ax.set_xlim(0,450)
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/ma_cost.png')
ma_gr = ma_subdf.groupby('Booking Year')
ma2015_df = ma_gr.get_group(2015)
ma2016_df = ma_gr.get_group(2016)

## N of columns
print (ch2015_df.shape[1]) #17
## N of different chain ID
print(len(ch2015_df.groupby(['Chain ID']).groups.keys())) #39
print(len(lv2015_df.groupby(['Chain ID']).groups.keys())) #31
print(len(ma2015_df.groupby(['Chain ID']).groups.keys())) #53

## adding a new column with the Chain name
ch2015_df.insert(17,'Chain Name',ch2015_df['Chain ID'].map(chains_df.set_index('Chain ID')['Chain Name']))
print(len(ch2015_df.groupby(['Chain Name']).groups.keys()))
ch2016_df.insert(17,'Chain Name',ch2016_df['Chain ID'].map(chains_df.set_index('Chain ID')['Chain Name']))
print(len(ch2016_df.groupby(['Chain Name']).groups.keys()))

lv2015_df.insert(17,'Chain Name',lv2015_df['Chain ID'].map(chains_df.set_index('Chain ID')['Chain Name']))
print(len(lv2015_df.groupby(['Chain Name']).groups.keys()))
lv2016_df.insert(17,'Chain Name',lv2016_df['Chain ID'].map(chains_df.set_index('Chain ID')['Chain Name']))
print(len(lv2016_df.groupby(['Chain Name']).groups.keys()))

ma2015_df.insert(17,'Chain Name',ma2015_df['Chain ID'].map(chains_df.set_index('Chain ID')['Chain Name']))
print(len(ma2015_df.groupby(['Chain Name']).groups.keys()))
ma2016_df.insert(17,'Chain Name',ma2016_df['Chain ID'].map(chains_df.set_index('Chain ID')['Chain Name']))
print(len(ma2016_df.groupby(['Chain Name']).groups.keys()))



##### CHICAGO ##### 
ch_2015_revSum = ch2015_df.groupby('Chain Name')['Revenue'].sum()
ch_2015_revMean = ch2015_df.groupby('Chain Name')['Revenue'].mean()
ch_2015_profSum = ch2015_df.groupby('Chain Name')['Profit'].sum()
ch_2015_profMean = ch2015_df.groupby('Chain Name')['Profit'].mean()
ch_2015_roomSum = ch2015_df.groupby('Chain Name')['RoomN'].sum()
ch_2015_roomMean = ch2015_df.groupby('Chain Name')['RoomN'].mean()
ch_2015_shopMean = ch2015_df.groupby('Chain Name')['Shopper'].mean()
ch_2015_shopSum = ch2015_df.groupby('Chain Name')['Shopper'].sum()

Nbookings_ch_2015 = ch2015_df.groupby('Chain Name')['Booking_ID'].count()

ch_2016_revSum = ch2016_df.groupby('Chain Name')['Revenue'].sum()
ch_2016_revMean = ch2016_df.groupby('Chain Name')['Revenue'].mean()
ch_2016_profSum = ch2016_df.groupby('Chain Name')['Profit'].sum()
ch_2016_profMean = ch2016_df.groupby('Chain Name')['Profit'].mean()
ch_2016_roomSum = ch2016_df.groupby('Chain Name')['RoomN'].sum()
ch_2016_roomMean = ch2016_df.groupby('Chain Name')['RoomN'].mean()
ch_2016_shopMean = ch2016_df.groupby('Chain Name')['Shopper'].mean()
ch_2016_shopSum = ch2016_df.groupby('Chain Name')['Shopper'].sum()

Nbookings_ch_2016 = ch2016_df.groupby('Chain Name')['Booking_ID'].count()


plt.figure()    
(ch_2016_revSum/ch_2015_revSum).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/revSum_ch_yoy.png')
plt.figure()    
(ch_2016_revMean/ch_2015_revMean).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/revMean_ch_yoy.png')
plt.figure()    
(ch_2016_profSum/ch_2015_profSum).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/profSum_ch_yoy.png')
plt.figure()    
(ch_2016_profMean/ch_2015_profMean).plot(figsize=(12,4),marker='o')
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/profMean_ch_yoy.png')
plt.figure()    
(ch_2016_roomSum/ch_2015_roomSum).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/roomSum_ch_yoy.png')
plt.figure()    
(ch_2016_roomMean/ch_2015_roomMean).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/roomMean_ch_yoy.png')
plt.figure()    
(ch_2016_shopMean/ch_2015_shopMean).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/shopMean_ch_yoy.png')
plt.figure()    
(ch_2016_shopSum/ch_2015_shopSum).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/shopSum_ch_yoy.png')
plt.figure()
(Nbookings_ch_2016/Nbookings_ch_2015).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/Nbookings_ch_yoy.png')

plt.figure()
ch2016_df.plot(x='Booking Window Days',y='Profit',kind='scatter')
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/prof_window_ch_2016.png')
plt.figure()
ch2016_df.plot(x='Booking Window Days',y='Revenue',kind='scatter')
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/rev_window_ch_2016.png')


##### LAS VEGAS ##### 
lv_2015_revSum = lv2015_df.groupby('Chain Name')['Revenue'].sum()
lv_2015_revMean = lv2015_df.groupby('Chain Name')['Revenue'].mean()
lv_2015_profSum = lv2015_df.groupby('Chain Name')['Profit'].sum()
lv_2015_profMean = lv2015_df.groupby('Chain Name')['Profit'].mean()
lv_2015_roomSum = lv2015_df.groupby('Chain Name')['RoomN'].sum()
lv_2015_roomMean = lv2015_df.groupby('Chain Name')['RoomN'].mean()
lv_2015_shopMean = lv2015_df.groupby('Chain Name')['Shopper'].mean()
lv_2015_shopSum = lv2015_df.groupby('Chain Name')['Shopper'].sum()

Nbookings_lv_2015 = lv2015_df.groupby('Chain Name')['Booking_ID'].count()

lv_2016_revSum = lv2016_df.groupby('Chain Name')['Revenue'].sum()
lv_2016_revMean = lv2016_df.groupby('Chain Name')['Revenue'].mean()
lv_2016_profSum = lv2016_df.groupby('Chain Name')['Profit'].sum()
lv_2016_profMean = lv2016_df.groupby('Chain Name')['Profit'].mean()
lv_2016_roomSum = lv2016_df.groupby('Chain Name')['RoomN'].sum()
lv_2016_roomMean = lv2016_df.groupby('Chain Name')['RoomN'].mean()
lv_2016_shopMean = lv2016_df.groupby('Chain Name')['Shopper'].mean()
lv_2016_shopSum = lv2016_df.groupby('Chain Name')['Shopper'].sum()

Nbookings_lv_2016 = lv2016_df.groupby('Chain Name')['Booking_ID'].count()


plt.figure()    
(lv_2016_revSum/lv_2015_revSum).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/revSum_lv_yoy.png')
plt.figure()    
(lv_2016_revMean/lv_2015_revMean).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/revMean_lv_yoy.png')
plt.figure()    
(lv_2016_profSum/lv_2015_profSum).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/profSum_lv_yoy.png')
plt.figure()    
(lv_2016_profMean/lv_2015_profMean).plot(figsize=(12,4),marker='o')
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/profMean_lv_yoy.png')
plt.figure()    
(lv_2016_roomSum/lv_2015_roomSum).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/roomSum_lv_yoy.png')
plt.figure()    
(lv_2016_roomMean/lv_2015_roomMean).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/roomMean_lv_yoy.png')
plt.figure()    
(lv_2016_shopMean/lv_2015_shopMean).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/shopMean_lv_yoy.png')
plt.figure()    
(lv_2016_shopSum/lv_2015_shopSum).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/shopSum_lv_yoy.png')
plt.figure()
(Nbookings_lv_2016/Nbookings_lv_2015).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/Nbookings_lv_yoy.png')

plt.figure()
lv2016_df.plot(x='Booking Window Days',y='Profit',kind='scatter')
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/prof_window_lv_2016.png')
plt.figure()
lv2016_df.plot(x='Booking Window Days',y='Revenue',kind='scatter')
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/rev_window_lv_2016.png')


##### MANHATTAN ##### 
ma_2015_revSum = ma2015_df.groupby('Chain Name')['Revenue'].sum()
ma_2015_revMean = ma2015_df.groupby('Chain Name')['Revenue'].mean()
ma_2015_profSum = ma2015_df.groupby('Chain Name')['Profit'].sum()
ma_2015_profMean = ma2015_df.groupby('Chain Name')['Profit'].mean()
ma_2015_roomSum = ma2015_df.groupby('Chain Name')['RoomN'].sum()
ma_2015_roomMean = ma2015_df.groupby('Chain Name')['RoomN'].mean()
ma_2015_shopMean = ma2015_df.groupby('Chain Name')['Shopper'].mean()
ma_2015_shopSum = ma2015_df.groupby('Chain Name')['Shopper'].sum()

Nbookings_ma_2015 = ma2015_df.groupby('Chain Name')['Booking_ID'].count()

ma_2016_revSum = ma2016_df.groupby('Chain Name')['Revenue'].sum()
ma_2016_revMean = ma2016_df.groupby('Chain Name')['Revenue'].mean()
ma_2016_profSum = ma2016_df.groupby('Chain Name')['Profit'].sum()
ma_2016_profMean = ma2016_df.groupby('Chain Name')['Profit'].mean()
ma_2016_roomSum = ma2016_df.groupby('Chain Name')['RoomN'].sum()
ma_2016_roomMean = ma2016_df.groupby('Chain Name')['RoomN'].mean()
ma_2016_shopMean = ma2016_df.groupby('Chain Name')['Shopper'].mean()
ma_2016_shopSum = ma2016_df.groupby('Chain Name')['Shopper'].sum()

Nbookings_ma_2016 = ma2016_df.groupby('Chain Name')['Booking_ID'].count()


plt.figure()    
(ma_2016_revSum/ma_2015_revSum).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/revSum_ma_yoy.png')
plt.figure()    
(ma_2016_revMean/ma_2015_revMean).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/revMean_ma_yoy.png')
plt.figure()    
(ma_2016_profSum/ma_2015_profSum).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/profSum_ma_yoy.png')
plt.figure()    
(ma_2016_profMean/ma_2015_profMean).plot(figsize=(12,4),marker='o')
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/profMean_ma_yoy.png')
plt.figure()    
(ma_2016_roomSum/ma_2015_roomSum).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/roomSum_ma_yoy.png')
plt.figure()    
(ma_2016_roomMean/ma_2015_roomMean).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/roomMean_ma_yoy.png')
plt.figure()    
(ma_2016_shopMean/ma_2015_shopMean).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/shopMean_ma_yoy.png')
plt.figure()    
(ma_2016_shopSum/ma_2015_shopSum).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/shopSum_ma_yoy.png')
plt.figure()
(Nbookings_ma_2016/Nbookings_ma_2015).plot(figsize=(12,4))
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/Nbookings_ma_yoy.png')

plt.figure()
ma2016_df.plot(x='Booking Window Days',y='Profit',kind='scatter')
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/prof_window_ma_2016.png')
plt.figure()
ma2016_df.plot(x='Booking Window Days',y='Revenue',kind='scatter')
plt.savefig('/Users/giulia/Applications/Jobs/Expedia_march2022/task_0523/plots/rev_window_ma_2016.png')

## some plotting for Chicago
## 2015

plt.figure()
ch2015_df.groupby('Chain Name')['Profit'].sum().plot(figsize=(12,4))
plt.figure()
ch2015_df.groupby('Chain Name')['Revenue'].sum().plot(figsize=(12,4))
plt.figure()
ch2015_df.groupby('Chain Name')['RoomN'].sum().plot(figsize=(12,4))
plt.figure()
ch2015_df.groupby('Chain Name')['RoomN'].mean().plot(figsize=(12,4))
plt.figure()
ch2015_df.groupby('Chain Name')['Booking Window Days'].mean().plot(figsize=(12,4))
plt.figure()
ch2015_df.plot(x='Booking Window Days',y='Profit',kind='scatter')
plt.figure()
ch2015_df.plot(x='Booking Window Days',y='Revenue',kind='scatter')


x = ch2015_df['Book Date']
y = ch2015_df.groupby('Chain Name')['Revenue'].mean()
plt.figure()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.plot(x,y)
plt.gcf().autofmt_xdate()

# 2016
plt.figure()
ch2016_df.groupby('Chain Name')['Profit'].sum().plot(figsize=(12,4))
plt.figure()
ch2016_df.groupby('Chain Name')['Revenue'].sum().plot(figsize=(12,4))
plt.figure()
ch2016_df.groupby('Chain Name')['RoomN'].sum().plot(figsize=(12,4))
plt.figure()
ch2016_df.groupby('Chain Name')['RoomN'].mean().plot(figsize=(12,4))
plt.figure()
ch2016_df.groupby('Chain Name')['Booking Window Days'].mean().plot(figsize=(12,4))



## some plotting for LV
## 2015
plt.figure()
lv2015_df.groupby('Chain Name')['Profit'].sum().plot(figsize=(12,4))
plt.figure()
lv2015_df.groupby('Chain Name')['Revenue'].sum().plot(figsize=(12,4))
plt.figure()
lv2015_df.groupby('Chain Name')['RoomN'].sum().plot(figsize=(12,4))
plt.figure()
lv2015_df.groupby('Chain Name')['RoomN'].mean().plot(figsize=(12,4))
plt.figure()
lv2015_df.groupby('Chain Name')['Booking Window Days'].mean().plot(figsize=(12,4))
plt.figure()
lv2015_df.plot(x='Booking Window Days',y='Profit',kind='scatter')
plt.figure()
lv2015_df.plot(x='Booking Window Days',y='Revenue',kind='scatter')


## 2016
plt.figure()
lv2016_df.groupby('Chain Name')['Profit'].sum().plot(figsize=(12,4))
plt.figure()
lv2016_df.groupby('Chain Name')['Revenue'].sum().plot(figsize=(12,4))
plt.figure()
lv2016_df.groupby('Chain Name')['RoomN'].sum().plot(figsize=(12,4))
plt.figure()
lv2016_df.groupby('Chain Name')['RoomN'].mean().plot(figsize=(12,4))
plt.figure()
lv2016_df.groupby('Chain Name')['Booking Window Days'].mean().plot(figsize=(12,4))
plt.figure()
lv2016_df.plot(x='Booking Window Days',y='Profit',kind='scatter')
plt.figure()
lv2016_df.plot(x='Booking Window Days',y='Revenue',kind='scatter')


## some plotting for Manhattan
## 2015
plt.figure()
ma2015_df.groupby('Chain Name')['Profit'].sum().plot(figsize=(12,4))
plt.figure()
ma2015_df.groupby('Chain Name')['Revenue'].sum().plot(figsize=(12,4))
plt.figure()
ma2015_df.groupby('Chain Name')['RoomN'].sum().plot(figsize=(12,4))
plt.figure()
ma2015_df.groupby('Chain Name')['RoomN'].mean().plot(figsize=(12,4))
plt.figure()
ma2015_df.groupby('Chain Name')['Booking Window Days'].mean().plot(figsize=(12,4))
plt.figure()
ma2015_df.plot(x='Booking Window Days',y='Profit',kind='scatter')
plt.figure()
ma2015_df.plot(x='Booking Window Days',y='Revenue',kind='scatter')


## 2016
plt.figure()
ma2016_df.groupby('Chain Name')['Profit'].sum().plot(figsize=(12,4))
plt.figure()
ma2016_df.groupby('Chain Name')['Revenue'].sum().plot(figsize=(12,4))
plt.figure()
ma2016_df.groupby('Chain Name')['RoomN'].sum().plot(figsize=(12,4))
plt.figure()
ma2016_df.groupby('Chain Name')['RoomN'].mean().plot(figsize=(12,4))
plt.figure()
ma2016_df.groupby('Chain Name')['Booking Window Days'].mean().plot(figsize=(12,4))
plt.figure()
ma2016_df.plot(x='Booking Window Days',y='Profit',kind='scatter')
plt.figure()
ma2016_df.plot(x='Booking Window Days',y='Revenue',kind='scatter')

## Shoppers analysis
plt.figure()
ma2016_df.groupby('Chain Name')['Shopper'].sum().plot(figsize=(12,4))
plt.figure()
ma2016_df.groupby('Chain Name')['Shopper'].mean().plot(figsize=(12,4))


### Manhattan
## define mask
mask_pre = (ma2016_df['Book Date'] >= '2016-10-14') & (ma2016_df['Book Date'] < '2016-10-28')
pre_ihg = ma2016_df.loc[mask_pre].groupby('Chain Name').get_group('IHG')
mask_post = (ma2016_df['Book Date'] >= '2016-10-28') & (ma2016_df['Book Date'] <= '2016-11-11')
post_ihg = ma2016_df.loc[mask_post].groupby('Chain Name').get_group('IHG')

pre_all = ma2016_df.loc[mask_pre]
#pre_all = pre_all.loc[pre_all['Chain Name'] != 'IHG']
post_all = ma2016_df.loc[mask_post]
#post_all = post_all.loc[post_all['Chain Name'] != 'IHG']

pre_allRev = pre_all['Revenue'].sum()
post_allRev = post_all['Revenue'].sum()
print(pre_allRev) 
print(post_allRev) 

pre_ihgRev = pre_ihg['Revenue'].sum()
post_ihgRev = post_ihg['Revenue'].sum()
print(pre_ihgRev) 
print(post_ihgRev) 

share_pre = pre_ihgRev/pre_allRev
print(share_pre)
share_post = post_ihgRev/post_allRev
print(share_post)

ratio_share = share_post/share_pre
print(ratio_share)

#share pre: 0.042
#share post: 0.061
# ratio = 1.45 -> 45% increase in performance

pre_shopper = pre_ihg['Shopper'].sum()
print(pre_shopper)
post_shopper = post_ihg['Shopper'].sum()
print(post_shopper)
print(post_shopper/pre_shopper)

## for Marriot
mask_pre = (ma2016_df['Book Date'] >= '2016-10-14') & (ma2016_df['Book Date'] < '2016-10-28')
pre_mar = ma2016_df.loc[mask_pre].groupby('Chain Name').get_group('Marriot')
mask_post = (ma2016_df['Book Date'] >= '2016-10-28') & (ma2016_df['Book Date'] <= '2016-11-11')
post_mar = ma2016_df.loc[mask_post].groupby('Chain Name').get_group('Marriot')

mar = ma2016_df.groupby('Chain Name').get_group('Marriott')
rev_mar = mar['Revenue'].sum()
print(rev_mar)


plt.figure()
pre = pre_ihg.groupby('Chain Name')['Shopper'].mean()
#.plot(figsize=(12,4))
post = post_ihg.groupby('Chain Name')['Shopper'].mean()
#.plot(figsize=(12,4))
ratio = post.divide(pre)
print(ratio)
ratio = ratio[ratio >=1]
print(ratio)
ratio.plot(figsize=(20,4))

## Marriot : 1.015

## Same with Chicago
plt.figure()
ch2016_df.groupby('Chain Name')['Shopper'].sum().plot(figsize=(12,4))
plt.figure()
ch2016_df.groupby('Chain Name')['Shopper'].mean().plot(figsize=(12,4))

## define mask
mask_pre_ch = (ch2016_df['Book Date'] >= '2016-10-14') & (ch2016_df['Book Date'] < '2016-10-28')
pre_ihg_ch = ch2016_df.loc[mask_pre_ch]
mask_post_ch = (ch2016_df['Book Date'] >= '2016-10-28') & (ch2016_df['Book Date'] <= '2016-11-11')
post_ihg_ch = ch2016_df.loc[mask_post_ch]
plt.figure()
pre_ch = pre_ihg_ch.groupby('Chain Name')['Shopper'].mean()
#.plot(figsize=(12,4))
post_ch = post_ihg_ch.groupby('Chain Name')['Shopper'].mean()
#.plot(figsize=(12,4))
ratio_ch = post_ch.divide(pre_ch)
print(ratio_ch)
ratio_ch = ratio_ch[ratio_ch >=1]
print(ratio_ch)
ratio_ch.plot(figsize=(20,4))







