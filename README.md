# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 13-05-2026
# NAME : YUVARAJ M
# REG NO : 212224040377

### AIM:
To implement ARMA model in python.
### Tools required:
```
Stock market data set ,
google colab
```
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.

### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

FILE_PATH = "/mnt/data/dataset(1).xlsx"

df = pd.read_excel(FILE_PATH)

print(df.head())

df = df[['Date', 'Open']].dropna()

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values('Date')

df['YearMonth'] = df['Date'].dt.to_period('M')

monthly = df.groupby('YearMonth')['Open'].mean().reset_index()

monthly.index = monthly['YearMonth'].dt.to_timestamp()

X = monthly['Open']

print("Data points :", len(X))
print(monthly.head())

plt.rcParams['figure.figsize'] = [12, 6]

plt.plot(X)
plt.title('Original Monthly Average Open Price')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.show()

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plot_acf(X, lags=40, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=40, ax=plt.gca(), method='ywm')
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

print("\nARMA(1,1) Model Summary")
print(arma11_model.summary())

phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

N = 1000

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.figure(figsize=(12, 6))
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1)')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_1)
plt.title('ACF of ARMA(1,1)')
plt.show()

plot_pacf(ARMA_1, method='ywm')
plt.title('PACF of ARMA(1,1)')
plt.show()

arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

print("\nARMA(2,2) Model Summary")
print(arma22_model.summary())

phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']

theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])

ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 10)

plt.figure(figsize=(12, 6))
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2)')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2)
plt.title('ACF of ARMA(2,2)')
plt.show()

plot_pacf(ARMA_2, method='ywm')
plt.title('PACF of ARMA(2,2)')
plt.show()
```
OUTPUT:
<img width="810" height="707" alt="image" src="https://github.com/user-attachments/assets/97c1eff1-a48e-43a8-a5c1-2af7de3f222f" />
<img width="823" height="715" alt="image" src="https://github.com/user-attachments/assets/4ac97ea6-ac81-4a59-988a-54756db0072e" />
<img width="803" height="719" alt="image" src="https://github.com/user-attachments/assets/91fbc0e4-e670-4ce8-b2c6-a03a58b22777" />
<img width="805" height="823" alt="image" src="https://github.com/user-attachments/assets/0bf16f05-5b36-4e08-b476-1b3f6464d008" />
<img width="825" height="438" alt="image" src="https://github.com/user-attachments/assets/dbbdb4e0-b536-4cfc-910b-4bb53e1565a2" />




## RESULT:
Thus, a python program is created to fir ARMA Model successfully.
