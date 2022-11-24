#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yfinance as yf
from scipy.stats import norm





df = pd.read_csv('us-shareprices-daily.csv',sep=';')
df.head()
#df.to_csv('df.csv')
#df.reset_index(inplace=True)
#display(df)




sp500 = yf.download (tickers = "^GSPC", start = "2016-10-24",end = "2021-10-23", interval = "1d")
sp500 = sp500['Adj Close']
sp500 = pd.DataFrame(sp500)
sp500.rename(columns={'Adj Close': 'sp500'}, inplace=True)
sp500.head(2)




#sp500.reset_index(inplace=True)
#sp500





stocks_name = df['Ticker'].unique()
#stocks_name
df_by_stock= df.pivot(index='Date',columns='Ticker')
df_by_stock



# dropping SimFinID, Dividend, and Shares Outstanding columns
df_by_stock = df_by_stock.drop(['SimFinId','Dividend','Shares Outstanding'],axis=1)





# merging stocks with sp500 data
data = df_by_stock['Adj. Close']
dd =pd.merge(data, sp500, left_index=True, right_index=True)
#dd.tail(15)





# reseting index for DataTime index
dd.reset_index(inplace=True)
dd.to_csv('dd.csv')

display(dd)




class CAPM(object):
    def __init__(self,tickers_quantity,dataset):
        self.dataset = dataset
    
    def tickers_return(self,tickers_quantity):
        tickers = []
        quantity = []
        for key,values in tickers_quantity.items():
            tickers.append(key)
            quantity.append(values)
        tickers.append('sp500')
        selected_stocks=self.dataset[tickers]
        ## calculating daily return
        # loops through each stocks
        # loops through each row belonging to the stock
        # calculates the percentage change from previous day
        # sets the value of first row to zero since there is no previous value
        df_stocks =selected_stocks.copy()
        for i in selected_stocks.columns[1:]:
            for j in range(1, len(selected_stocks)):
                df_stocks[i][j] = ((selected_stocks[i][j]- selected_stocks[i][j-1])/selected_stocks[i][j-1]) * 100
            df_stocks[i][0] = 0
        # calculate Beta and alpha for a single stock
        # used sp500 as a benchmark
        # used polyfit to calculate beta
        beta={}
        alpha={}
        stocks_daily_return = df_stocks
        print(df_stocks)
        
        for i in stocks_daily_return.columns:
            if i != 'Date' and i != 'sp500':
                #stocks_daily_return.plot(kind = 'scatter', x = 'A', y = i)
                b, a = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return[i], 1)
                #plt.plot(stocks_daily_return['sp500'], b * stocks_daily_return['sp500'] + a, '-', color = 'r')
                beta[i] = round(b,2)
                alpha[i] = round(a,2)
                #plt.show()
                
        #calculating camp for a stock        
        keys = list(beta.keys())
        ER= {}
        # rf = 0 assuming risk-free rate of 0
        rf =0
        # rm - annualize retun
        rm = stocks_daily_return['sp500'].mean()*252
        for i in keys:
            ER[i] = round(rf + (beta[i]*(rm-rf)),2)
            
        for i in keys:
            print('Expected Return based on CAPM for {} is {}%'.format(i,ER[i]))
        print(ER)
        
        import matplotlib.pyplot as plt
        plt.bar(*zip(*ER.items()))
        plt.plot(rm)
        plt.show()
        
        # calculate expected return for the portfolio
        # portfolio weights assume equal
        portfolio_weights =[]
        current_cash_value =0
        total_portfolio_value = 0
        for i in range(len(tickers)-1):
            stocks_name = tickers[i]
            current_cash_value = selected_stocks[stocks_name].iloc[-1]
            stocks_quantity = quantity[i]
            cash_value = stocks_quantity*current_cash_value
            total_portfolio_value += cash_value
            portfolio_weights.append(cash_value)
        print(portfolio_weights)
        portfolio_weights = (portfolio_weights/total_portfolio_value)*100            
        #portfolio_weights =np.array(portfolio_weights)
        print(portfolio_weights)
        
        
        #portfolio_weights = 1/(len(tickers)-1) * np.ones(len(tickers)-1)
        ER_portfolio = round(sum(list(ER.values()) * portfolio_weights)/100,2)
        print('Expected Return Bases on CAPM for the portfolio is {}%\n'.format(ER_portfolio))
        result = {'beta': beta,
                  'alpha':alpha,
                  'ER':ER,
                  'ER_portfolio':ER_portfolio}
        return result




tickers_quantity={'AA':25,'GOOG':50,'A':100}
capm = CAPM(tickers_quantity,dd)
capm.tickers_return(tickers_quantity)







