# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:06:35 2023

@author: MLAND01
"""
import numpy as np
import pandas as pd 
import time

from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

def set_step_dates(report_1):
    '''
    adds step column to report_1 with list of dates of steps  

    Parameters
    ----------
    report_1 : TYPE
        DESCRIPTION.

    Returns
    -------
    report_1 : TYPE
        DESCRIPTION.

   '''
    for row in report_1.itertuples():
        dates=row[3]
        if len(dates)>0:
            dates = np.array(dates, dtype='datetime64')
            diffs = np.diff(dates)  # Calculate the difference between consecutive dates
            sequences = []   # Initialize list to store sequences
            start = dates[0]        # Initialize start and end of sequence
            end = dates[0]        
            for i in range(len(diffs)): # Iterate over differences
               if diffs[i] == np.timedelta64(1, 'D'):
                   end = dates[i+1] # If dates are consecutive, update end of sequence
               else:
                   # If dates are not consecutive, add current sequence to list and start a new sequence
                   sequences.append((start, end))
                   start = dates[i+1]
                   end = dates[i+1]
            sequences.append((start, end)) # Add the last sequence to the list
         
            lst=[]
            for i in range(len(sequences)) :
                #print('i',i,sequences[i][0])
                
                lst.append(sequences[i][0])
            ind=row[0]
            if len(lst)>0:
                lst=pd.to_datetime(lst).strftime("%Y-%m-%d").tolist()
                #print(lst)
                report_1.at[ind,'steps']=lst
    return report_1

def set_delta_values(report_1,df):
    '''
    adds delta column to report_1 with delta values before and after each step

    Parameters
    ----------
    report_1 : TYPE
        DESCRIPTION.
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    report_1 : TYPE
        DESCRIPTION.

    '''
    for row in report_1.itertuples():
        cell=row[1]
        kpi=row[2]
        date=row[4]
        delta=[]
        for i in range(0,len(date)):
            pre=df[(df['ECELL_ID']==cell)&(df['Data']<date[i])][kpi].mean()
            post=df[(df['ECELL_ID']==cell)&(df['Data']>date[i])][kpi].mean()
            delta.append(post-pre)
        ind=row[0]
        report_1.at[ind,'delta']=delta
    return report_1

def set_last_value(report_1,df,date):
   '''
   adds last value column to report_1 with mean values from date to end  

    Parameters
    ----------
    report_1 : TYPE
        DESCRIPTION.
    df : TYPE
        DESCRIPTION.
    date : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

   '''   
   for row in report_1.itertuples():
        ind=row[0]
        cell=row[1]
        kpi=row[2]
        last=df[(df['ECELL_ID']==cell)&(df['Data']>date)][kpi].mean()
        report_1.at[ind,'last_value']=last
   return report_1 


def freq(report,number=None):
    '''
    reports a dataframe with columns 'kpi' and 'most_common'
    listing for each kpi the most common dates for anomalies
    to be used to check for generalized anomalies in a single day 
    number defines the number of anomalous days to be reported, none=all 
    
    Parameters
    ----------
    report : TYPE
        DESCRIPTION.

    Returns
    -------
    output : TYPE
        DESCRIPTION.

    '''
    kpis=list(report['KPI'].unique())
    output=pd.DataFrame(columns=['kpi','most_common'])
    for kpi in kpis :
        lst=[]
        df=report[report['KPI']==kpi].reset_index()
        for i in range(len(df['BKPT'])):
            tmp=df['BKPT'][i]
            for j in range(len(tmp)):
                lst.append(tmp[j])
        counter=Counter(lst)
        ordered=sorted(counter.items())
        x,y=[],[]
        for i in range(len(ordered)):
            x.append(ordered[i][0])
            y.append(ordered[i][1])
        plt.figure(figsize=(4, 2.5))
        plt.bar(x,y)
        plt.title(kpi)
        plt.xticks(rotation='vertical')
        row_to_append=pd.DataFrame({'kpi': kpi, 'most_common' : counter.most_common(number)})
        output=pd.concat([output,row_to_append])
        
    return output

def Trend(df,window_start,window_end,slope=1,debug=False):
      '''
      reports a dataframe with columns CELLID,KPI,TREND 
      for every cell and every kpi trend column states if the series has a positive , stable or negative trend
      slope introduces a minimum threshold not to be considered as stable
      window start and window end can limit the area of investigation (e.g last 15 days) 
      
      
      '''
      timestart = time.time()
      print('trend start')
      cells=df['ECELL_ID'].unique() # list of cells available in file 
     # inserire controlli su window start e winfdow end 
      #window_start=df_end-pd.Timedelta(days=15)
      df_trend=pd.DataFrame(columns=['CELLID','KPI','TREND'])
      for cell in cells:
            df_cell=df[(df['ECELL_ID']==cell)&(df['Data']>=window_start)&(df['Data']<=window_end)].drop('ECELL_ID',axis=1)# df_cell is a dataframe for a single cell
            if df_cell.shape[0] > 3 : #df_cell.set_index("Data", inplace=True)
                for i in range(1,len(df_cell.columns)): #loops on all columns (i.e KPIs) of df_cell
                    kpi=df_cell.columns[i]
                    if debug : print('trend function, cell=',cell, 'kpi=',kpi)
                    X = df_cell.index.values.reshape(-1,1)
                    y = df_cell.iloc[:,i].values 
                    # X=df.index.values.reshape(-1,1)
                    reg = LinearRegression().fit(X, y)               
                    calculated_slope = reg.coef_[0] # Get the slope of the trend line
                    if calculated_slope > slope :
                       trend = 'upward'
                    elif calculated_slope < -slope :
                       trend = ' downward'
                    else :
                       trend ='stable'
                    #present=report[(report['CELLID']==cell)&(report['KPI']==kpi)]['KPI'].any()
                    
                    #if present :
                    row_to_append=pd.DataFrame({'CELLID': cell, 'KPI' : kpi, 'TREND': trend},index=[0]) 
                    df_trend=pd.concat([df_trend,row_to_append])
                    #    row=report[(report['CELLID']==cell)&(report['KPI']==kpi)].index.item()
                    #   col=report.columns.get_loc('trend')
                    #    report.iloc[row,col]=trend
                    
                    if debug : 
                        # Get the prediction
                        y_pred = reg.predict(X)
                        # Get the fitting metrics
                        mape = mean_absolute_error(y, y_pred)
                        r2 = r2_score(y,y_pred)
                        mean = np.mean(y)
                        print('slope,mape,mean,r2', calculated_slope,mape,mean,r2)         
      timeend = time.time()
      print('elapsed seconds: ', timeend - timestart) 
      return df_trend 

def plot_rtwp(df,report):
    '''
    plots cells where steps have been identified and higlights them
    plot may be put into a ppt 
    Parameters
    ----------
    df : 
    report : 

    Returns
    -------
    None.

    '''
    timestart = time.time()
    print('plot_rtwp start')
    cells=report.index.to_list()
    kpis=report.columns.levels[0].to_list()
    kpis.remove('CELLID')
    #metrics=report.columns.level[1]
    for cell in cells:
        df_cell=df[df['ECELL_ID']==cell].drop('ECELL_ID',axis=1)# df_cell is a dataframe for a single cell
        df_cell.reset_index(inplace=True,drop=True)
        fig,ax=plt.subplots(len(kpis),1,figsize=(10,6))
        fig.suptitle(cell)
        i=0
        for kpi in kpis:
            if len(kpis)==1 : 
                s=sns.lineplot(x=df_cell.iloc[:,0], y=df_cell[kpi],ax=ax)
            else : 
                s=sns.lineplot(x=df_cell.iloc[:,0], y=df_cell[kpi],ax=ax[i])
            steps=report.filter(items = [cell],axis=0)[(kpi,'steps')].to_list()
            try :
                len(steps[0])
                for j in range(len(steps[0])):
                    bkdate=steps[0][j]
                    s.vlines(x=dt.datetime.strptime(bkdate, "%Y-%m-%d").date(),ymin=s.axes.get_ylim()[0],ymax=s.axes.get_ylim()[1],color='r')
                i=i+1
            except:
                i=i+1
                continue
            
        plt.tight_layout()
        '''           
        bkpts=report[(report['CELLID']==cell)]['steps'].reset_index(drop=True)
        print('cell,bkpts',cell,bkpts)
        lenghts=0
        for i in range(0,len(bkpts)):
            lenghts=lenghts+len(bkpts[i])
            print('lenghts',lenghts,i)
        if lenghts > 0 :
            fig, ax = plt.subplots(len(df_cell.columns)-1,1,figsize=(16,20),sharex=True)
            plt.suptitle(cell)
            #kpis=report[report['CELLID']==cell]['KPI']
            a=0
            for i in range(1,len(df_cell.columns)):
                print('i,a',i,a)
                kpi=df_cell.columns[i]
                #if (lenghts == 1) : 
                s=sns.lineplot(x=df_cell.iloc[:,0], y=df_cell[kpi])
                
                #   print('kpi',kpi)
                breakpoints=report[(report['CELLID']==cell)]['steps'].values
                #print('bkpts',bkpts,'len', len(bkpts))
                if len(breakpoints) > 0 :   
                    for j in range(0,len(breakpoints)):
                      #      print(j,bkpts[j])
                            bkdate=np.array(df_cell[df_cell['Data'].dt.strftime('%Y-%m-%d') == breakpoints[0][0]]['Data'])
                            s.vlines(x=bkdate,ymin=s.axes.get_ylim()[0],ymax=s.axes.get_ylim()[1],color='r')
                a=a+1
          '''
    del df_cell
    timeend = time.time()
    print('elapsed seconds: ', timeend - timestart)    
    return
def report_delta(report,delta):
    output=report.copy()
    kpis=output.columns.levels[0]
    indexes=[]
    for kpi in kpis:
        
        for ind in output[(kpi,'delta')].index :
            if abs(max(output[(kpi,'delta')][ind])) > delta :
                indexes.append(ind)
    indexes=list(set(indexes)) # remove duplicates in case of multiple kpis
    output=output.reset_index()
    output=output[output['CELLID'].isin(indexes)]
    output=output.set_index('CELLID')
    return output
        
    