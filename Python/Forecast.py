
#python ../scripts/forecastWAY_beauty_weekly.py -importType sql -mode forecast -table WAY_Forecast_InputDati -promozioni promozioni -importDate Dcd
#Tabella di input: dbo.WAY_Forecast_InputDati
#Tabella promozioni WAY_Forecast_Promozioni
#Creare la tabella di output con la seguente query:
#create table WAY_Forecast_Vendite (FCPV nchar(100), FCItemID nchar(100), FCDate date, FCValue decimal(18,4),FCValueHigh decimal(18,4), FCValueLow decimal(18,4), FCTitle nchar(100), FCComment nchar(100), FCCalculationDate date)
#Col1: colonna addizionale,Col2: Punto Vendita,col3: Oggetto, Data, Valore
import os
import pandas as pd
import numpy as np
import sys
from fbprophet import Prophet
from datetime import timedelta,datetime
import pyodbc
import argparse
import matplotlib.pyplot as plt
from fbprophet.diagnostics import cross_validation,performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import calendar
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
import logging
from dateutil.easter import *

#import custom modules
#from Utilities import utilities

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

#print start hour
i = datetime.now()
print ("Current hour Start= ",'{}:{}:{}'.format(i.hour,i.minute,i.second))  

#suppress prophet output (from pySTAN)
class suppress_stdout_stderr(object):

    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])



def forecast_decadi_next(df,max_date,TabellaDecadi,promozioni,col,ogg, pv):

    '''funzione per predire le vendite
    successive al periodo disponibile
    risultato per ogni settimana
    funzione per valutare la bonta' 
    dell'algoritmo'''

    res=df[['Decade','Valore']]
    print(res.head())
    res.columns=['ds','y']

    if not promozioni.empty:
        model = Prophet(interval_width=0.9,changepoints=list(res['ds']),holidays=promozioni)
    if promozioni.empty:
        model = Prophet(interval_width=0.9,changepoints=list(res['ds']))
    if max(res.ds)<max_date:
        res=res.append({'ds' : max_date, 'y' : 0} , ignore_index=True)
    with suppress_stdout_stderr():
        model.fit(res)
        future = model.make_future_dataframe(periods=120,freq='D')
        forecast = model.predict(future)

        model.plot(forecast)
        plt.savefig('{}{}{}{}'.format('forecast',pv,ogg,'.png'),bbox_inches='tight')
        plt.close()
        model.plot_components(forecast)
        plt.savefig('{}{}{}{}'.format('componenti',pv,ogg,'.png'),bbox_inches='tight')
        plt.close()
        #print(forecast)
    if col=='Dcd':
        tmp=pd.merge(forecast,TabellaDecadi,how='left',left_on='ds',right_on='COD_DAY')
        #fbprophet ripete il valore predetto su tutte le decadi, quindi sommo e divido per il numero dei giorni facenti parte della decade
        ngg=tmp.groupby(['Decade'])['yhat'].size().replace(1,10).reset_index()
        ngg=ngg.rename(columns={'yhat':'ngg'})
        #print(ngg)
        forecast=tmp.groupby(['Decade'])[['yhat','yhat_lower','yhat_upper']].mean().reset_index() 
        #print("A:\n",forecast.head())
        forecast['yhat_lower']=forecast['yhat_lower']
        forecast['yhat_upper']=forecast['yhat_upper']
        forecast[['yhat','yhat_lower','yhat_upper']]=forecast[['yhat','yhat_lower','yhat_upper']].divide(ngg['ngg'],axis=0)
        #print("B:\n",forecast.head())
        forecast=forecast[['yhat','yhat_lower','yhat_upper','Decade']]
        forecast.loc[:,'Decade']=pd.to_datetime(forecast.loc[:,'Decade'])
        result=pd.merge(forecast,res,left_on='Decade',right_on='ds')
        result=result[result.Decade<max_date].reset_index(drop=True)
        forecast=forecast[forecast.Decade>=max_date].reset_index(drop=True)
        #print(forecast)

    return forecast,result


def arrotonda(data):

    '''arrotonda i dati tenendo conto
        dello scarto'''
              
    data['tmp']=data['yhat'].round()
    data['resto']=data['yhat']-data['tmp']
    lista_resto=list()
    data['new_yhat']=data['tmp']
    data=data.reset_index(drop=True)
    resto=0.0
    for index in data.index:
        data.loc[index,'new_yhat']=(data.loc[index,'yhat']+resto).round()
        resto=(data.loc[index,'yhat']+resto)-data.loc[index,'new_yhat']
        lista_resto.append(resto)
    data['resto']=lista_resto
    del data['yhat']
    data=data.rename(columns={"new_yhat":"yhat"})
    return data

if __name__=="__main__":
    titolo = 'V3' #?
    commento = 'InputDecadi' 
    parser = argparse.ArgumentParser()
    parser.add_argument('-importType', help='sql per database or csv per file separati da ;') #sql
    parser.add_argument('-table',help='Nome della tabella di input') #input table
    parser.add_argument('-mode',help='forecast per prevedere, evaluation per valutare l\'algoritmo negli ultimi 120 gg') #forecast
    parser.add_argument('-max_date',help='parametro interno, si riferisce alla data di aggiornamento dei dati di input')
    parser.add_argument('-DateType',help='Dcd per Decadi, D per Giorni, W per Settimane, M per Mesi') #Giorni D, Settimane W, Decadi Dcd, Mesi M, Anni Y
    parser.add_argument('-pv',help='Punto Vendita')
    parser.add_argument('-promozioni',help='Aggiungere dataframe promozioni, nel formato puntovendita,Item,data')
    parser.add_argument('-SERVER',help='MSSQL Host, indirizzo server')
    parser.add_argument('-DataBASE',help='MSSQL DB, Nome del DataBase')
    parser.add_argument('-UID',help='MSSQL User, Nome Utente')
    parser.add_argument('-PWD',help='MSSQL Pwd, Password')
    parser.add_argument('-PORT',help='MSSQL PORT')
    args = parser.parse_args()
    print("script is running with the following parameters: ",args)
    table=args.table
    print(args.max_date)
    max_date=datetime.strptime(args.max_date,"%Y-%m-%d")

    pv=args.pv
    col=args.DateType
    print(pv)
    # Importo i dati
    if args.importType=='sql':
        cmd='{};{}={};{}={};{}={};{}={};{}={}'.format('DRIVER={SQL Server}','SERVER',args.SERVER,'PORT',args.PORT,'DataBASE',args.DataBASE,'UID',args.UID,'PWD',args.PWD)
        sqlConnection = pyodbc.connect(cmd)
        cursor = sqlConnection.cursor()
        sql='{} {} {} \'{}\''.format('SELECT * FROM',table,'WHERE Col2 = ',pv)
        #sql=(" Select * FROM WAY_Forecast_InputDati WHERE Col3 in ('117769','ARM05887') and Col2= "+"'"+pv+"'")
        
        print('query: ',sql)
        #print('query: ',sql)
        df_import = pd.read_sql(sql,sqlConnection)
        if args.promozioni=='True':
            sql=("SELECT * FROM [WAY_Forecast_Promozioni] WHERE PV = "+"'"+pv+"'")
            #print(sql)
            promozioni=pd.read_sql(sql,sqlConnection)
            promozioni = pd.DataFrame({'holiday': 'promozioni','ds': pd.to_datetime(['2017-12-21', '2018-01-01']),'lower_window': 0,'upper_window': 1})
            #print(promozioni)

        if args.promozioni=='False':
            promozioni=pd.DataFrame()
        
        #importo Tabella Decadi

        sql=("SELECT * FROM AN_PERIODI_DECADI")
        TabellaDecadi=pd.read_sql(sql,sqlConnection)
      

    

    df_import.loc[:,'Data'] = pd.to_datetime(df_import.loc[:,'Data'])  
    # Rinomino le colonne e ne imposto il tipo di dati
    df_import=df_import.rename(columns={"Col2":"PuntoVendita","Col3":"Oggetto"})

    
    if args.DateType=='D':

        #input in giorni

        TabellaDecadi.loc[:,'COD_DAY']=pd.to_datetime(TabellaDecadi.loc[:,'COD_DAY'])
        df_import=pd.merge(df_import,TabellaDecadi,how='left',left_on='Data',right_on='COD_DAY').reset_index(drop=True)
        df_import.loc[:,'COD_DAY']=pd.to_datetime(df_import.loc[:,'COD_DAY'])    
        #max_date=pd.to_datetime(TabellaDecadi.loc[TabellaDecadi['COD_DAY']==max_date,'COD_DAY'].unique()[0])
        #print(max_date)
        lista_ogg=df_import.Oggetto.unique()
        df_input=df_import[['PuntoVendita','Data','COD_DAY','Valore','Oggetto']].reset_index(drop=True)
        all_res= list()
        final_fc = pd.DataFrame()
        exception=list()
        #df_input=df_input.reset_index()
        #df_input=df_input.groupby(['PuntoVendita','Oggetto','COD_DAY'])['Valore'].sum().reset_index()
        #print(df_input)
        df_input=df_input.rename(columns={"COD_DAY":"Decade"})
        #print(df_input)

    if args.DateType=='Dcd':

        #input in decadi

        TabellaDecadi.loc[:,'COD_DAY']=pd.to_datetime(TabellaDecadi.loc[:,'COD_DAY'])
        df_import=pd.merge(df_import,TabellaDecadi,how='left',left_on='Data',right_on='COD_DAY').reset_index(drop=True)
        df_import.loc[:,'Decade']=pd.to_datetime(df_import.loc[:,'Decade'])    
        max_date=pd.to_datetime(TabellaDecadi.loc[TabellaDecadi['COD_DAY']==max_date,'Decade'].unique()[0])
        #print(max_date)
        lista_ogg=df_import.Oggetto.unique()
        df_input=df_import[['PuntoVendita','Data','Decade','Valore','Oggetto']].drop_duplicates().reset_index(drop=True)
        all_res= list()
        final_fc = pd.DataFrame()
        exception=list()
        df_input=df_input.groupby(['PuntoVendita','Oggetto','Decade'])['Valore'].sum().reset_index()

    if args.DateType=='M':

        #input in mesi

        TabellaDecadi.loc[:,'COD_DAY']=pd.to_datetime(TabellaDecadi.loc[:,'COD_DAY'])
        df_import=pd.merge(df_import,TabellaDecadi,how='left',left_on='Data',right_on='COD_MONTH').reset_index(drop=True)
        df_import.loc[:,'COD_MONTH']=pd.to_datetime(df_import.loc[:,'Decade'])    
        max_date=pd.to_datetime(TabellaDecadi.loc[TabellaDecadi['COD_DAY']==max_date,'COD_MONTH'].unique()[0])
        #print(max_date)
        lista_ogg=df_import.Oggetto.unique()
        df_input=df_import[['PuntoVendita','Data','COD_MONTH','Valore','Oggetto']].drop_duplicates().reset_index(drop=True)
        all_res= list()
        final_fc = pd.DataFrame()
        exception=list()
        df_input=df_input.groupby(['PuntoVendita','Oggetto','COD_MONTH'])['Valore'].sum().reset_index()

    #print('df_input',df_input.head())
    for ogg in df_input.Oggetto.unique():
        print(ogg)
        df=df_input[df_input.Oggetto==ogg].reset_index(drop=True)
        #print(df)
        if args.mode=='forecast':
            try:
                forecast,result=forecast_decadi_next(df,max_date,TabellaDecadi,promozioni,col,ogg,pv)
                #print(forecast.head())
                #controllo previsioni. Se la previsione è maggiore del 35% rispetto al max negli ultimi 2 anni correggo la previsione con il massimo del venduto
                forecast['Oggetto']=ogg
                forecast['PuntoVendita']=pv
                all_res.append(forecast)

            except Exception as e:
                logging.debug(e)
                logging.info('Ogg',ogg)
                logging.warning('PuntoVendita:',pv)
                pass
    if len(all_res)==0:
        all_res_next=pd.DataFrame.from_dict({"yhat":[0],"yhat_lower":[0],"yhat_upper":[0],"Decade":[0],"ds":[0],"y":[0],"Oggetto":[0],"PuntoVendita":[0]})
    if len(all_res)>0:
        all_res_next=pd.concat(all_res)

    
    if  all_res_next.shape[0]>=1:
        #arrotondo i dati in modo da tener conto del resto. Aggiungo 1 quando la somma dei resti risulta 0.5 
        #all_res_next["yhat"]=arrotonda(all_res_next)
        #all_res_next=arrotonda(all_res_next)
        all_res_next=all_res_next[['PuntoVendita','Oggetto', 'Decade', 'yhat', 'yhat_upper','yhat_lower']]
        print("scrittura su db..",all_res_next.shape)
        #print(all_res_next)
        #all_res_next=all_res_next.rename(columns={"new_yhat":"yhat"})
        all_res_next.loc[all_res_next['yhat']<0,'yhat']=abs(0)
        all_res_next.loc[all_res_next.yhat_lower<0,'yhat_lower']=abs(0)
        all_res_next.loc[all_res_next.yhat_upper<0,'yhat_upper']=abs(0)
        all_res_next = all_res_next[~np.all(all_res_next == 0, axis=1)]
        all_res_next['FCCalculationDate']=datetime.today().strftime("%y-%m-%dT%H:%M:%S")
        all_res_next['titolo'] = titolo
        all_res_next['commento'] = commento
        all_res_next=all_res_next[['PuntoVendita','Oggetto', 'Decade', 'yhat', 'yhat_upper','yhat_lower', 'titolo', 'commento','FCCalculationDate']]
        all_res_next.to_csv('{}_{}'.format(pv,"RISULTATI.csv"),sep=";",header=True,index=None)  
        all_res_next['Oggetto']=all_res_next['Oggetto'].str.strip()
        if args.importType=='sql':
            print("scrittura su db..",all_res_next.shape[0])
            all_res_next=all_res_next.reset_index(drop=True)
            all_res_next['Oggetto']=all_res_next['Oggetto'].str.strip()

            for index, row in all_res_next.iterrows():
                print(index)
                try:
                    cursor.execute(
                        "INSERT INTO WAY_Forecast_Vendite_PVART (FCPV, FCItemID, FCDate, FCValue, FCValueHigh,FCValueLow, FCTitle, FCComment, FCCalculationDate) VALUES (?,?, ?, ?, ?, ?, ?, ?, GETDATE())",
                        (row['PuntoVendita'], row['Oggetto'], row['Decade'], row['yhat'], row['yhat_upper'], row['yhat_lower'], row['titolo'],
                        row['commento']))

                except Exception as e:
                    print("errore")
                    print(e)
        
            sqlConnection.commit()


    

        