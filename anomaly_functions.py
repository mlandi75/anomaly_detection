# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:06:35 2023

@author: MLAND01
"""
import numpy as np
import pandas as pd 
import time
from adtk.visualization import plot
from adtk.data import validate_series
#from adtk.detector import PersistAD
from adtk.detector import LevelShiftAD

from scipy import signal
from prophet import Prophet

import scipy.stats as stats

import logging

def init(df) : 
    #try:
    #    del anomalies_matrix
    #except:
    #    pass
    #dtype=pd.SparseDtype(float,fill_value=0)
    cols=df.columns[2:]
    newcols=[]
    for i in range(len(cols)):
        newcols.append(cols[i]+'_LS')
        newcols.append(cols[i]+'_PE')
    newcols.append('Anomalies_count')
    #cols=cols.append(pd.Index(['Anomalies_count']))
    tmp=df[['ECELL_ID','Data']].copy()
    ind=pd.MultiIndex.from_frame(tmp)
    anomalies_matrix=pd.DataFrame(np.nan,index=ind,columns=newcols)
    #anomalies_matrix=anomalies_matrix.astype(dtype)
    del tmp
    #anomalies_matrix['ECELL_ID']=df['ECELL_ID'].copy()
    #anomalies_matrix['Data']=df['Data'].copy()
   
    print('initialized anomalies_matrix')
    return anomalies_matrix

def anomalies_adtk_ls(df,cells,kpis,anomalies_matrix,c=3.0,window=5,side='positive',graph=False,debug=False):
    '''   
    anomaly detection function based on ADTK LevelShiftAD
    Parameters
    ----------
    df : dataframe with data 
    cells : list of cells to work on 
    kpis : list of kpis to work on 
    anomalies_matrix : needed to return updated anomalies matrix - not used 
    c : (float, optional) – Factor used to determine the bound of normal range based on historical interquartile range.The default is 3.0.
    window :  (int or str, or 2-tuple of int or str) 
            Size of the time windows.
            If int, it is the number of time point in this time window.
            If str, it must be able to be converted into a pandas Timedelta object.
            If 2-tuple, it defines the left and right window respectively.
            The default is 5.
    side :  (str, optional) 
        If “both”, to detect anomalous positive and negative changes;
        If “positive”, to only detect anomalous positive changes;
        If “negative”, to only detect anomalous negative changes.
        The default is 'positive'.
    graph : flag to enable plot di TUTTE le timeseries
        The default is False.
    debug : flag to enable some print screen
        The default is False.

    Returns
    -------
    error : flag true if at elast one cell not processed found (and creates a csv file - TO BE IMPROVED)
    output : report file with list of breakpoints
    anomalies matrix updated (not used) 

    '''
    timestart = time.time()
    print('anomalies_adtk_ls start')
    error=False
    output=pd.DataFrame(columns=['CELLID','KPI','BKPT'])
    for cell in cells:
        df_cell=df[df['ECELL_ID']==cell]
        if debug: print(cell)
        start,end=df_cell['Data'].min(),df_cell['Data'].max()
# ricreo range perche non piace ad adtk 
        new_ind=pd.date_range(start=start, end=end, freq='D') 
        df_cell.index=df_cell['Data']
        df_cell.reindex(new_ind)
        df_cell=df_cell.drop('Data',axis=1)
        for kpi in kpis:
            try:
                s = validate_series(df_cell[kpi])
                level_shift_ad = LevelShiftAD(c=c, side=side, window=window)
                anomalies = level_shift_ad.fit_detect(s)
                if graph :
                    ax=plot(s, anomaly=anomalies, anomaly_color='red');
                    ax[0].set_title('LS'+str(cell))
                    
                bkpts=anomalies[anomalies==True].index.values
              
                anomalies_matrix.loc[(cell,bkpts),kpi]=1.0
                lst=[]
                if len(bkpts)>0:
                    dates=pd.to_datetime(bkpts).strftime("%Y-%m-%d").tolist()
                    lst.append([cell,kpi,dates])
                    output=pd.concat([output, pd.DataFrame(lst,columns=['CELLID','KPI','BKPT'])], ignore_index = True)
            except:
                filename=str(cell)+'.csv'
                df_cell.to_csv(filename)
                error=True
    timeend = time.time()
    print('elapsed seconds: ', timeend - timestart)    
    return error , output, anomalies_matrix

def gradino_scipy(df,cells,kpis,anomalies_matrix,prominence=20,width=(10,20),debug=False):
    '''
    anomaly detection function based on convolution with heaviside function and scipy signal find peaks function
    
    Parameters
    ----------
    df : dataframe with data 
    cells : list of cells to work on 
    kpis : list of kpis to work on 
    anomalies_matrix : needed to return updated anomalies matrix - not used
    prominence : threshold for peak prominance in order to be detected, the higher the less peaks are detected
        The default is 20.
    width : defines width of the peak
        The default is (10,20).
    debug : flag to enable some print screen
        The default is False.

    Returns
    -------
    error : flag true if at elast one cell not processed found (and creates a csv file - TO BE IMPROVED)
    output : report file with list of breakpoints
    anomalies matrix updated (not used) 

    '''
    timestart = time.time()
    print('gradino_scipy start')
    error=False
    output=pd.DataFrame(columns=['CELLID','KPI','BKPT'])
    for cell in cells:
        if debug : print('gradino_scipy, cell=',cell)
        df_cell=df[df['ECELL_ID']==cell].drop('ECELL_ID',axis=1)# df_cell is a dataframe for a single cell
        if df_cell.shape[0]<=2:
            name=str(cell)+'.csv'
            df_cell.to_csv(name)
            break
        for kpi in kpis: #loops on all columns (i.e KPIs) of df_cell
            col=df_cell[kpi]
            dary=np.array(df_cell[kpi])
            dary -= np.average(dary)
            step = np.hstack((np.ones(len(dary)), -1*np.ones(len(dary))))
            dary_pos_steps = np.convolve(dary, step, mode='valid')
        #    step_indx = np.argmax(dary_step) 
            pos_peaks = signal.find_peaks(dary_pos_steps,prominence=20,width=(10,20))[0]
            dary_neg_steps = np.convolve(-dary, step, mode='valid')
            neg_peaks = signal.find_peaks(dary_neg_steps,prominence=20,width=(10,20))[0]
            bkposdates=np.array(df_cell.iloc[pos_peaks,0]) #.index.to_pydatetime(), dtype='datetime64[D]')
            bknegdates=np.array(df_cell.iloc[neg_peaks,0]) #.index.to_pydatetime(), dtype='datetime64[D]')
            bkpts=np.concatenate((bkposdates,bknegdates), axis=None)
            #anomalies_matrix.loc[(cell,bkposdates),col]=1.0
            #anomalies_matrix.loc[(cell,bknegdates),col]=1.0
            anomalies_matrix.loc[(cell,bkpts),kpi]=1.0
            lst=[]
            if len(bkpts)>0:
                dates=pd.to_datetime(bkpts).strftime("%Y-%m-%d").tolist()
                lst.append([cell,kpi,dates])
                output=pd.concat([output, pd.DataFrame(lst,columns=['CELLID','KPI','BKPT'])], ignore_index = True)
                
        del df_cell  
    timeend = time.time()
    print('elapsed seconds: ', timeend - timestart)    
    return error , output, anomalies_matrix

def anomaly_prophet(df,cells,kpis,anomalies_matrix,width=0.89,delta_perc=None,debug=False):
    '''
    Function trains a forecast model based on prophet and detects as anomalies all values outside confidence interval
    Better for peaks or last days anomalies rather than steps 
    Parameters
    ----------
    df : dataframe with data 
    cells : list of cells to work on 
    kpis : list of kpis to work on 
    anomalies_matrix : needed to return updated anomalies matrix - not used
    width : determines the shape of confidence interval.It is the % of the model confidence 
        The default is 0.89.
    delta_perc : threshold to filter anomalies. 
        Are considered only the anomalies whose distance from the average exeeds threshold (expressed in %)
        The default is None.
    debug : flag to enable some print screen
        The default is False.

    Returns
    -------
    error : flag true if at elast one cell not processed found (and creates a csv file - TO BE IMPROVED)
    output : report file with list of breakpoints
    anomalies matrix updated (not used) 

    '''
    timestart = time.time()
    print('anomaly_prophet start')
    error=False    
    logger = logging.getLogger('cmdstanpy')
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)
    output=pd.DataFrame(columns=['CELLID','KPI','BKPT'])
    for cell in cells:
        if debug : print('anomaly_prophet_rate, cell=',cell)
        df_cell=df[df['ECELL_ID']==cell].drop('ECELL_ID',axis=1)# df_cell is a dataframe for a single cell
        #df_cell.set_index('Data',inplace=True)
        if df_cell.shape[0]<=2:
            name=str(cell)+'.csv'
            df_cell.to_csv(name)
        else :
            for kpi in kpis: #loops on all columns (i.e KPIs) of df_cell
                data=pd.DataFrame(columns=['ds','y'])
                
                
                data['ds']=df_cell.iloc[:,0]
                data['ds'] = pd.to_datetime(data['ds'])
                data['y']=df_cell[kpi] #prepares data in the format requested by Prophet
            
                model = Prophet(interval_width=width, weekly_seasonality=True) #to be added seasonality as further input
                model.fit(data)     # Fit the model on the training dataset
                forecast = model.predict(data) # Make prediction
                performance = pd.merge(data, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')     # Merge actual and predicted values
                performance['anomaly'] = performance.apply(lambda rows: 1 if ((rows.y>rows.yhat_upper)) else 0, axis = 1) # Create an anomaly indicator; being a rate only positive spikes are considered
                #performance['anomaly'] = performance.apply(lambda rows: 1 if ((rows.y<rows.yhat_lower)|(rows.y>rows.yhat_upper)) else 0, axis = 1)
               
                if delta_perc :      # if delta_perc then keeps only anomalies above threshold  
                    avg=performance['y'].mean()
                    performance['delta_perc']=abs((performance['y']-avg)/avg)
                    mask=((performance['anomaly']==1) & (performance['delta_perc']<delta_perc))
                    performance.loc[mask,'anomaly']=0
        
                bkpts=performance[performance['anomaly']==1].index.values # lists indexes of all breakpoints
                bkpts=np.array(df_cell.iloc[bkpts,0]) #.index.to_pydatetime(), dtype='datetime64[D]')
                anomalies_matrix.loc[(cell,bkpts),kpi]=1.0
                lst=[]
                if len(bkpts)>0:
                    dates=pd.to_datetime(bkpts).strftime("%Y-%m-%d").tolist()
                    lst.append([cell,kpi,dates])
                    output=pd.concat([output, pd.DataFrame(lst,columns=['CELLID','KPI','BKPT'])], ignore_index = True)
                    
                
                if debug : # Check MAE and MAPE value for performance check if needed
                    performance_MAE = mean_absolute_error(performance['y'], performance['yhat'])
                    performance_MAPE = mean_absolute_percentage_error(performance['y'], performance['yhat'])
                    s1= str(cell) + ' MAE='+ str(round(performance_MAE,2)) +' MAPE='+str(round(performance_MAPE,2))
                    tmp=performance['anomaly'].value_counts()
                    try:
                        s2 ='anomalies count =' + str(tmp[1])
                    except:
                        s2 ='anomalies count = 0'
                    print('performance indicators for', s1)
                    print(s2)
                       
                del performance,data
        del df_cell
        timeend = time.time()
        print('elapsed seconds: ', timeend - timestart)      
    return error , output, anomalies_matrix

def anomaly_z_score(df, cells,kpis,anomalies_matrix,threshold=3,debug=False):
    '''
    Function detects anoalies based on Z_score calculation 
    Better for peaks or last days anomalies rather than steps 
    Parameters
    ----------
    df : dataframe with data 
    cells : list of cells to work on 
    kpis : list of kpis to work on 
    anomalies_matrix : needed to return updated anomalies matrix - not used
    threshold : Z-score threshold to detect anomalies 
        The default is 3.
    debug : flag to enable some print screen
        The default is False.

    Returns
    -------
   error : flag true if at elast one cell not processed found (and creates a csv file - TO BE IMPROVED)
   output : report file with list of breakpoints
   anomalies matrix updated (not used) 

    '''
    error=False
    output=pd.DataFrame(columns=['CELLID','KPI','BKPT'])
    for cell in cells:
        if debug : print('anomaly_z_score, cell=',cell)
        df_cell=df[df['ECELL_ID']==cell].drop('ECELL_ID',axis=1)# df_cell is a dataframe for a single cell
        df_cell.index=df_cell['Data']
        df_cell.drop('Data',axis=1,inplace=True)
        if df_cell.shape[0]<=2:
            name=str(cell)+'.csv'
            df_cell.to_csv(name)
        else :
            for kpi in kpis: #loops on all columns (i.e KPIs) of df_cell
                ts=df_cell[kpi].to_numpy()
                scores=pd.DataFrame()
                scores['z-score']=stats.zscore(ts)
                bkpts=scores.loc[scores['z-score']>threshold].index.tolist()
                if debug: print('bkpts',bkpts)
                if len(bkpts)>0:
                    i=df_cell.columns.get_loc(kpi)
                    mia=np.array(df_cell.iloc[bkpts,i].index.to_pydatetime(), dtype='datetime64[D]')
                    bkpts=mia
                    anomalies_matrix.loc[(cell,bkpts),kpi]=1.0
                    lst=[]
                
                    dates=pd.to_datetime(bkpts).strftime("%Y-%m-%d").tolist()
                    lst.append([cell,kpi,dates])
                    output=pd.concat([output, pd.DataFrame(lst,columns=['CELLID','KPI','BKPT'])], ignore_index = True)
        del df_cell
                         
    return error , output, anomalies_matrix
 