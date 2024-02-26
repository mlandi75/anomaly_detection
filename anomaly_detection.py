# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:03:31 2023

@author: MLAND01
Detects anomalies in time series input data using adtk library and also scipy library 
(anomaly prohpet and z score not used due too many anomaly idetifications - more fit for spikes rather than steps)
data file format :  file format .csv, separator ";", columns: ECELL_ID;Data;KPI1;KPI2; ... KPIn;DayOfWeek;
for input parameters see settings part 
creates report 1 'CELLID', 'KPI', 'BKPT', 'steps', 'delta', 'last_value'
creates anomalies matrix.csv but for future use 
creates report2.csv with 3 columns for each KPI , identified steps, delta and last value , with ECELLID as index
plots the anomalies found (plot_rtwp)
plots the frequency of anomalies per day (output)
calculates trend for each KPI based on input window (trend) - not used 

set path for input file !!!
"""
import numpy as np
import pandas as pd
import os

# anomaly_functions.py contains anomaly detection algorythms 
from anomaly_functions import init, anomalies_adtk_ls,gradino_scipy,anomaly_prophet,anomaly_z_score
# anomaly_report_functions.py contains function to create reports and plot results  
from anomaly_report_functions import plot_rtwp,set_step_dates, set_delta_values,set_last_value,freq,Trend,report_delta

#%%adtk_settings
c=6.0
window=3
side='both'
graph=False
#gradino_scipy settings
prominence=20
width=(10,20)
#anomaly_prophet settings
width_anomaly=0.89
delta_perc=None
#anomaly_z_score settings
threshold=3

#%% Data import 

#path=r'C:\Users\mland01\OneDrive - Vodafone Group\Documenti\Jupyterfiles\timeseries'
path=os.getcwd()
filename=os.path.join(path,'rtwp_gradino5.csv')
#filename=os.path.join(path,'BLER_DL_5G_MI.csv')
df = pd.read_csv(filename,delimiter=";")
df['Data']=pd.to_datetime(df['Data'],format="%d/%m/%Y %H:%M:%S")
if 'CELL_ID' in df.columns :
    df.rename(columns={'CELL_ID' : 'ECELL_ID'},inplace=True)
df.drop('DayOfWeek',axis=1,inplace=True)
df = df.iloc[: , :-1] #removes DayofWeek column
print('Data imported')
#%% Data cleaning AGGIUNGERE FILTERING CONDITIONS 
# filtrare righe per filtering condition e poi rimuovere le colonne che non siano kpi da valutare 
df.fillna(0,inplace=True)
cells_orig=df['ECELL_ID'].unique()
#df=df.iloc[0:120,:]

#%% Preliminary steps
anomalies_matrix=init(df)
cells=df['ECELL_ID'].unique()
kpis=df.columns[2:].values 

#%% fill anomalies matrix 
err_a,err_b,err_c,err_d = None,None,None,None
err_a,report_a,anomalies_matrix=anomalies_adtk_ls(df,cells,kpis,anomalies_matrix,c=c,window=window,side=side,graph=graph,debug=False)
report_1=report_a
err_b,report_b,anomalies_matrix=gradino_scipy(df,cells,kpis,anomalies_matrix,prominence=prominence,width=width,debug=False)
report_1=pd.merge(report_a,report_b,how='outer',on=['CELLID','KPI'])
report_1['BKPT_x'].fillna({i: [] for i in report_1.index},inplace=True)
report_1['BKPT_y'].fillna({i: [] for i in report_1.index},inplace=True)
report_1['BKPT']=report_1['BKPT_x']+report_1['BKPT_y']
report_1.drop(['BKPT_x','BKPT_y'],axis=1,inplace=True)
'''
err_c,report_c,anomalies_matrix=anomaly_prophet(df,cells,kpis,anomalies_matrix,width=width_anomaly,delta_perc=delta_perc,debug=False)
report_1=pd.merge(report_1,report_c,how='outer',on=['CELLID','KPI'])
report_1['BKPT_x'].fillna({i: [] for i in report_1.index},inplace=True)
report_1['BKPT_y'].fillna({i: [] for i in report_1.index},inplace=True)
report_1['BKPT']=report_1['BKPT_x']+report_1['BKPT_y']
report_1.drop(['BKPT_x','BKPT_y'],axis=1,inplace=True)
err_d,report_d,anomalies_matrix=anomaly_z_score(df, cells,kpis,anomalies_matrix,threshold=threshold,debug=False)
report_1=pd.merge(report_1,report_d,how='outer',on=['CELLID','KPI'])
report_1['BKPT_x'].fillna({i: [] for i in report_1.index},inplace=True)
report_1['BKPT_y'].fillna({i: [] for i in report_1.index},inplace=True)
report_1['BKPT']=report_1['BKPT_x']+report_1['BKPT_y']
report_1.drop(['BKPT_x','BKPT_y'],axis=1,inplace=True)
'''
report_1['BKPT']=report_1['BKPT'].apply(lambda x : np.unique(x))

# functions log a csv with problematic Cell_ID, to be improved
if (err_a or err_b or err_c or err_d) : print('trovata almeno una cella problematica') 
anomalies_matrix['Anomalies_count']=anomalies_matrix.count(axis=1)  
anomalies_matrix.to_csv('anomalies_matrix.csv')

#%% create report 
'''
report_1 list all bpkts and calculates delta and last value 
'''
report_1['steps']=np.nan
report_1['steps'] = report_1['steps'].astype('object')
report_1=set_step_dates(report_1)
report_1['delta']=0
report_1['delta'] = report_1['delta'].astype('object')
#%% # MIGIORABILE
report_1=set_delta_values(report_1,df) 
report_1['last_value']=0
date=df['Data'].max()-pd.Timedelta(days=2)
report_1=set_last_value(report_1,df,date)

report_1.to_csv('report_1.csv')
'''
report_2 lists the anomalies found with best date
'''
if report_1.shape[0]>0:
    report_2=report_1.drop('BKPT',axis=1).set_index(['CELLID','KPI'])
    #report_2=report_2[report_2['delta'].apply(np.min)>0] # delta da parametrizzare
    report_2=report_2.unstack('KPI')
    report_2.columns = report_2.columns.swaplevel()
    new_cols=[]
    for i in report_2.columns.levels[0]:
        for j in report_2.columns.levels[1]:
           new_cols.append((i,j))
    multi_cols = pd.MultiIndex.from_tuples(new_cols, names=['KPI', 'Info'])
    report_2 = pd.DataFrame(report_2, columns=multi_cols)
    report_2.to_csv('report_2.csv')
    
report_3=report_delta(report_2,2)
report_3.to_csv('report_3.csv')
plot_rtwp(df,report_3) # plots the anomalies found 

output = freq(report_1) # calculates freq of anomalies dates and plots  

#calculates trend for each KPI based on input window, to be used for future check 
window_end=df['Data'].unique().max()
#window_start=window_end-pd.Timedelta(days=15)
window_start=df['Data'].unique().min()
trend = Trend(df,window_start=window_start,window_end=window_end,slope=0.3,debug=False)
        
