import numpy as np
import sklearn
from matplotlib import pyplot as plt
import pandas as pd
import datetime
from pandas_datareader import data



# Define the instruments to download

companies_dict = {
	'Amazon': 'AMZN',
	'Apple': 'AAPL',
	'Walgreens': 'WBA',
	'Northop Grumman': 'NOC',
	'Boeing': 'BA',
	'Lockheed Martin': 'LMT',
	'McDonalds': 'MCD',
	'Intel': 'INTC',
	'Navistar': 'NAV',
	'IBM': 'IBM',
	'Texas Instruments': 'TXN',
	'MasterCard': 'MA',
	'Microsoft': 'MSFT',
	'General Electrics': 'GE',
	'Symantec': 'NLOK',
	'American Express': 'AXP',
	'Pepsi': 'PEP',
	'Coca Cola': 'KO',
	'Johnson & Johnson': 'JNJ',
	'Toyota': 'TM',
	'Honda': 'HMC',
	'Mitsubishi': 'MSBHF',
	'Sony': 'SNY',
	'Exxon': 'XOM',
	'Chevron': 'CVX',
	'Valero Energy': 'VLO',
	'Ford': 'F',
	'Bank of America': 'BAC'}

companies = sorted(companies_dict.items(), key = lambda x: x[1])

#print(companies)

data_source = 'yahoo'

# Start and end dates
start_date = '2015-01-01'
end_date = '2017-12-31'

# Use pandas_datareader to load the desired stock data
panel_data = data.DataReader(list(companies_dict.values()), data_source, start_date, end_date)

# Print Axes labels
#print(panel_data)

# Find Stock Close and Open data values
stock_close = panel_data.loc['Close']
stock_open = panel_data.loc['Open']

print(stock_close.iloc[100])